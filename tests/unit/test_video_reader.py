"""Lector de video (E0 §2) sobre un clip H.264 generado durante la prueba."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("av")

from precision_weeder.video.reader import (
    VideoDecodeError,
    backend_versions,
    decode_sequential,
    iter_frames,
    probe_video,
    sha256_file,
)
from precision_weeder.video.timing import analyze_timestamps


def test_probe_reports_declared_metadata(synthetic_video: Path) -> None:
    probe = probe_video(synthetic_video)
    assert probe.codec == "h264"
    assert (probe.width_px, probe.height_px) == (192, 128)
    assert probe.frame_count_reported == 24
    assert probe.fps_nominal == pytest.approx(15.0, abs=1e-9, rel=0)
    assert probe.fps_nominal_source.startswith("stream.")
    assert probe.time_base is not None
    assert probe.duration_reported_s == pytest.approx(1.6, abs=1e-6, rel=0)


def test_sequential_decode_recovers_count_order_and_timestamps(
    synthetic_video: Path, video_fixture_module
) -> None:
    order, times = [], []
    for frame in iter_frames(synthetic_video, image_format="bgr24"):
        order.append(video_fixture_module.decode_index(frame.image))
        times.append(frame.timestamp_s)
        assert frame.image.shape == (128, 192, 3)
    assert order == list(range(24))
    report = analyze_timestamps(times)
    assert report.source == "pts"
    assert report.fps_effective == pytest.approx(15.0, abs=1e-9, rel=0)
    assert report.monotonic_increasing and not report.anomalies


def test_decode_report_reaches_eof_without_warnings(synthetic_video: Path) -> None:
    frames, report = decode_sequential(synthetic_video)
    assert report.frames_decoded == len(frames) == 24
    assert report.eof_status == "eof_reached"
    assert report.observed_sizes == {(192, 128)}
    assert not report.truncation_suspected
    assert report.frames_without_pts == 0


def test_lossy_frames_are_close_to_the_source_not_identical(
    synthetic_video: Path, video_fixture_module
) -> None:
    """Tolerancia explicita del codec: error medio absoluto < 6 niveles de 255."""
    import numpy as np

    for frame in iter_frames(synthetic_video, image_format="bgr24"):
        expected = video_fixture_module.frame_image(frame.index, 4)
        error = np.abs(frame.image.astype(float) - expected.astype(float)).mean()
        assert error < 6.0


def test_pts_gaps_are_detected(tmp_path: Path, video_fixture_module) -> None:
    ticks = [*range(12), *range(14, 26)]
    path = video_fixture_module.write_video(tmp_path / "gap.mp4", pts_ticks=ticks)
    frames, _ = decode_sequential(path)
    report = analyze_timestamps([f.timestamp_s for f in frames])
    assert report.gaps == 1
    assert report.anomalies[0]["after_index"] == 11
    assert report.anomalies[0]["interval_s"] == pytest.approx(3 / 15, abs=1e-6, rel=0)


def test_missing_file_is_an_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(VideoDecodeError, match="inexistente"):
        probe_video(tmp_path / "no.mp4")
    with pytest.raises(VideoDecodeError, match="inexistente"):
        list(iter_frames(tmp_path / "no.mp4"))


def test_corrupt_file_is_an_explicit_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00not a video" * 400)
    with pytest.raises(VideoDecodeError, match="no se pudo abrir"):
        probe_video(corrupt)


def test_truncated_file_without_index_cannot_be_opened(
    tmp_path: Path, video_fixture_module
) -> None:
    path = video_fixture_module.write_video(tmp_path / "full.mp4")
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(path.read_bytes()[: int(path.stat().st_size * 0.6)])
    with pytest.raises(VideoDecodeError):
        probe_video(truncated)


def test_truncated_faststart_file_reports_decode_error(
    tmp_path: Path, video_fixture_module
) -> None:
    path = video_fixture_module.write_video(tmp_path / "fs.mp4", faststart=True)
    truncated = tmp_path / "fs_truncated.mp4"
    truncated.write_bytes(path.read_bytes()[: int(path.stat().st_size * 0.6)])
    probe = probe_video(truncated)
    _, report = decode_sequential(truncated)
    assert report.frames_decoded < (probe.frame_count_reported or 0)
    assert report.eof_status == "decode_error"
    assert report.truncation_suspected
    assert report.warnings  # el registro de FFmpeg se conserva, no se oculta
    assert "limitation" in report.to_dict()


def test_hash_and_backend_versions(synthetic_video: Path) -> None:
    assert len(sha256_file(synthetic_video)) == 64
    versions = backend_versions()
    assert versions["pyav"] and "libavcodec" in versions and versions["opencv"] != ""
