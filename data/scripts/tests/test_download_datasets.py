from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import download_datasets as dd


def test_compute_shortfall_when_below_target() -> None:
    assert dd.compute_shortfall(existing=120, target=200) == 80


def test_compute_shortfall_when_at_or_above_target() -> None:
    assert dd.compute_shortfall(existing=200, target=200) == 0
    assert dd.compute_shortfall(existing=250, target=200) == 0


def test_compute_shortfall_when_nothing_collected_yet() -> None:
    assert dd.compute_shortfall(existing=0, target=200) == 200


def test_count_existing_counts_only_image_files(tmp_path: Path) -> None:
    class_dir = tmp_path / "left_front"
    class_dir.mkdir()
    (class_dir / "a.jpg").write_bytes(b"fake")
    (class_dir / "b.PNG").write_bytes(b"fake")  # extension case should not matter
    (class_dir / "notes.txt").write_bytes(b"not an image")
    (class_dir / "subdir").mkdir()  # directories are not files

    assert dd.count_existing(class_dir) == 2


def test_count_existing_returns_zero_for_missing_dir(tmp_path: Path) -> None:
    assert dd.count_existing(tmp_path / "does-not-exist") == 0


def test_count_collected_sums_final_and_unsorted_dirs(tmp_path: Path, monkeypatch: Any) -> None:
    exterior_dir = tmp_path / "exterior"
    unsorted_dir = exterior_dir / "_unsorted"
    monkeypatch.setattr(dd, "EXTERIOR_DIR", exterior_dir)
    monkeypatch.setattr(dd, "UNSORTED_DIR", unsorted_dir)

    (exterior_dir / "left_front").mkdir(parents=True)
    (exterior_dir / "left_front" / "a.jpg").write_bytes(b"fake")
    (exterior_dir / "left_front" / "b.jpg").write_bytes(b"fake")
    (unsorted_dir / "left_front").mkdir(parents=True)
    (unsorted_dir / "left_front" / "c.jpg").write_bytes(b"fake")

    assert dd.count_collected("left_front") == 3


def test_count_collected_is_zero_when_neither_dir_exists(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(dd, "EXTERIOR_DIR", tmp_path / "exterior")
    monkeypatch.setattr(dd, "UNSORTED_DIR", tmp_path / "exterior" / "_unsorted")

    assert dd.count_collected("right_rear") == 0


def test_count_collected_counts_final_dir_only_when_unsorted_is_empty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    exterior_dir = tmp_path / "exterior"
    monkeypatch.setattr(dd, "EXTERIOR_DIR", exterior_dir)
    monkeypatch.setattr(dd, "UNSORTED_DIR", exterior_dir / "_unsorted")
    (exterior_dir / "right_front").mkdir(parents=True)
    (exterior_dir / "right_front" / "a.jpg").write_bytes(b"fake")

    assert dd.count_collected("right_front") == 1


def test_count_collected_counts_unsorted_dir_only_when_final_is_empty(
    tmp_path: Path, monkeypatch: Any
) -> None:
    exterior_dir = tmp_path / "exterior"
    unsorted_dir = exterior_dir / "_unsorted"
    monkeypatch.setattr(dd, "EXTERIOR_DIR", exterior_dir)
    monkeypatch.setattr(dd, "UNSORTED_DIR", unsorted_dir)
    (unsorted_dir / "right_front").mkdir(parents=True)
    (unsorted_dir / "right_front" / "a.jpg").write_bytes(b"fake")

    assert dd.count_collected("right_front") == 1


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_fetch_pexels_parses_candidates_without_network(monkeypatch: Any) -> None:
    payload = {
        "photos": [
            {
                "photographer": "Jane Doe",
                "url": "https://www.pexels.com/photo/123",
                "src": {"large": "https://images.pexels.com/photos/123/large.jpg"},
            },
            {
                # Missing "src.large": should be skipped, not crash.
                "photographer": "No Image",
                "url": "https://www.pexels.com/photo/456",
                "src": {},
            },
        ]
    }

    def fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(payload)

    monkeypatch.setattr(dd.requests, "get", fake_get)

    candidates = dd.fetch_pexels("car right front angle", api_key="fake-key")

    assert candidates == [
        dd.ImageCandidate(
            source="pexels",
            url="https://images.pexels.com/photos/123/large.jpg",
            source_url="https://www.pexels.com/photo/123",
            author="Jane Doe",
            license="Pexels License",
        )
    ]


def test_fetch_unsplash_parses_candidates_without_network(monkeypatch: Any) -> None:
    payload = {
        "results": [
            {
                "urls": {"regular": "https://images.unsplash.com/photo-1"},
                "links": {
                    "html": "https://unsplash.com/photos/1",
                    "download_location": "https://api.unsplash.com/photos/1/download",
                },
                "user": {"name": "Alex"},
            }
        ]
    }

    def fake_get(*_args: Any, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(payload)

    monkeypatch.setattr(dd.requests, "get", fake_get)

    candidates = dd.fetch_unsplash("car left rear angle", access_key="fake-key")

    assert candidates == [
        dd.ImageCandidate(
            source="unsplash",
            url="https://images.unsplash.com/photo-1",
            source_url="https://unsplash.com/photos/1",
            author="Alex",
            license="Unsplash License",
            download_location="https://api.unsplash.com/photos/1/download",
        )
    ]


class _FakeStatusResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_confirm_unsplash_download_true_on_200(monkeypatch: Any) -> None:
    calls = []

    def fake_get(url: str, **kwargs: Any) -> _FakeStatusResponse:
        calls.append((url, kwargs.get("headers")))
        return _FakeStatusResponse(200)

    monkeypatch.setattr(dd.requests, "get", fake_get)

    ok = dd.confirm_unsplash_download(
        "https://api.unsplash.com/photos/1/download", access_key="fake-key"
    )

    assert ok is True
    assert calls == [
        (
            "https://api.unsplash.com/photos/1/download",
            {"Authorization": "Client-ID fake-key"},
        )
    ]


def test_confirm_unsplash_download_false_on_non_200(monkeypatch: Any) -> None:
    monkeypatch.setattr(dd.requests, "get", lambda *a, **k: _FakeStatusResponse(403))

    ok = dd.confirm_unsplash_download(
        "https://api.unsplash.com/photos/1/download", access_key="fake-key"
    )

    assert ok is False


def test_confirm_unsplash_download_false_on_request_exception(monkeypatch: Any) -> None:
    def raise_error(*_args: Any, **_kwargs: Any) -> Any:
        raise dd.requests.RequestException("boom")

    monkeypatch.setattr(dd.requests, "get", raise_error)

    ok = dd.confirm_unsplash_download(
        "https://api.unsplash.com/photos/1/download", access_key="fake-key"
    )

    assert ok is False


class _FakeDownloadResponse:
    """Minimal stand-in for the `requests.get` response of an image download."""

    def __init__(self, content: bytes) -> None:
        self.status_code = 200
        self.headers = {"content-type": "image/jpeg"}
        self.content = content


def test_collect_confirms_unsplash_download_location_before_saving(
    tmp_path: Path, monkeypatch: Any
) -> None:
    unsorted_dir = tmp_path / "_unsorted"
    monkeypatch.setattr(dd, "UNSORTED_DIR", unsorted_dir)
    monkeypatch.setattr(dd, "MANIFEST_PATH", tmp_path / "manifest.csv")

    candidate = dd.ImageCandidate(
        source="unsplash",
        url="https://images.unsplash.com/photo-1",
        source_url="https://unsplash.com/photos/1",
        author="Alex",
        license="Unsplash License",
        download_location="https://api.unsplash.com/photos/1/download",
    )
    monkeypatch.setattr(dd, "fetch_unsplash", lambda *a, **k: [candidate])
    monkeypatch.setattr(dd, "fetch_pexels", lambda *a, **k: [])
    monkeypatch.setattr(dd, "fetch_pixabay", lambda *a, **k: [])

    calls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> Any:
        calls.append(url)
        if url == candidate.download_location:
            assert kwargs.get("headers") == {"Authorization": "Client-ID fake-key"}
            return _FakeStatusResponse(200)
        return _FakeDownloadResponse(b"fake jpeg bytes")

    monkeypatch.setattr(dd.requests, "get", fake_get)

    saved = dd.collect(
        "left_front",
        ["query"],
        target=1,
        keys={"unsplash": "fake-key", "pexels": None, "pixabay": None},
        seen_hashes=set(),
    )

    assert saved == 1
    assert candidate.download_location in calls
    # download_location must be confirmed before the image is written out.
    assert calls.index(candidate.download_location) < calls.index(candidate.url)
    saved_files = list((unsorted_dir / "left_front").glob("*.jpg"))
    assert len(saved_files) == 1


def test_collect_skips_image_when_download_location_confirmation_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    unsorted_dir = tmp_path / "_unsorted"
    monkeypatch.setattr(dd, "UNSORTED_DIR", unsorted_dir)
    monkeypatch.setattr(dd, "MANIFEST_PATH", tmp_path / "manifest.csv")

    candidate = dd.ImageCandidate(
        source="unsplash",
        url="https://images.unsplash.com/photo-1",
        source_url="https://unsplash.com/photos/1",
        author="Alex",
        license="Unsplash License",
        download_location="https://api.unsplash.com/photos/1/download",
    )
    monkeypatch.setattr(dd, "fetch_unsplash", lambda *a, **k: [candidate])
    monkeypatch.setattr(dd, "fetch_pexels", lambda *a, **k: [])
    monkeypatch.setattr(dd, "fetch_pixabay", lambda *a, **k: [])

    def fake_get(url: str, **_kwargs: Any) -> Any:
        if url == candidate.download_location:
            return _FakeStatusResponse(403)  # confirmation denied/failed
        return _FakeDownloadResponse(b"fake jpeg bytes")

    monkeypatch.setattr(dd.requests, "get", fake_get)

    saved = dd.collect(
        "left_front",
        ["query"],
        target=1,
        keys={"unsplash": "fake-key", "pexels": None, "pixabay": None},
        seen_hashes=set(),
    )

    assert saved == 0
    assert not (tmp_path / "manifest.csv").exists()
    saved_files = list((unsorted_dir / "left_front").glob("*.jpg"))
    assert saved_files == []


def test_append_manifest_row_writes_header_once(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    row = {
        "file": "pexels_abc123.jpg",
        "class": "left_front",
        "source": "pexels",
        "source_url": "https://example.com/photo",
        "author": "Jane Doe",
        "license": "Pexels License",
        "downloaded_at": "2026-01-01T00:00:00+00:00",
    }

    dd.append_manifest_row(manifest_path, row)
    dd.append_manifest_row(manifest_path, {**row, "file": "pexels_def456.jpg"})

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert [r["file"] for r in rows] == ["pexels_abc123.jpg", "pexels_def456.jpg"]
    assert rows[0]["class"] == "left_front"
