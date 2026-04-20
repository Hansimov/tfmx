from __future__ import annotations

"""Reusable ffmpeg-backed audio preprocessing profiles for QSR transcription."""

import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path


FFMPEG_AUDIO_PROFILE_REQUIRED_MESSAGE = (
    "ffmpeg is required for qsr audio profile rendering. Install it first, then retry."
)

SOURCE_AUDIO_PROFILE = "source"
MONO_MID_AUDIO_PROFILE = "mono_mid"
LEFT_ONLY_AUDIO_PROFILE = "left_only"
RIGHT_ONLY_AUDIO_PROFILE = "right_only"
MONO_MID_LOUDNORM_AUDIO_PROFILE = "mono_mid_loudnorm"
LEFT_ONLY_LOUDNORM_AUDIO_PROFILE = "left_only_loudnorm"
RIGHT_ONLY_LOUDNORM_AUDIO_PROFILE = "right_only_loudnorm"
VOICE_BAND_MID_AUDIO_PROFILE = "voice_band_mid"


@dataclass(frozen=True)
class AudioTransformProfile:
    name: str
    suffix: str | None = None
    filtergraph: str | None = None
    channels: int | None = None
    sample_rate: int | None = None
    codec: str | None = None
    bitrate: str | None = None
    movflags: str | None = None

    @property
    def requires_render(self) -> bool:
        return any(
            value is not None
            for value in (
                self.filtergraph,
                self.channels,
                self.sample_rate,
                self.codec,
                self.bitrate,
                self.movflags,
            )
        )


AUDIO_TRANSFORM_PROFILES: dict[str, AudioTransformProfile] = {
    SOURCE_AUDIO_PROFILE: AudioTransformProfile(
        name=SOURCE_AUDIO_PROFILE,
        suffix=None,
    ),
    MONO_MID_AUDIO_PROFILE: AudioTransformProfile(
        name=MONO_MID_AUDIO_PROFILE,
        suffix=".m4a",
        filtergraph="pan=mono|c0=0.5*c0+0.5*c1",
        channels=1,
        sample_rate=32000,
        codec="aac",
        bitrate="64k",
        movflags="+faststart",
    ),
    LEFT_ONLY_AUDIO_PROFILE: AudioTransformProfile(
        name=LEFT_ONLY_AUDIO_PROFILE,
        suffix=".m4a",
        filtergraph="pan=mono|c0=c0",
        channels=1,
        sample_rate=32000,
        codec="aac",
        bitrate="64k",
        movflags="+faststart",
    ),
    RIGHT_ONLY_AUDIO_PROFILE: AudioTransformProfile(
        name=RIGHT_ONLY_AUDIO_PROFILE,
        suffix=".m4a",
        filtergraph="pan=mono|c0=c1",
        channels=1,
        sample_rate=32000,
        codec="aac",
        bitrate="64k",
        movflags="+faststart",
    ),
    MONO_MID_LOUDNORM_AUDIO_PROFILE: AudioTransformProfile(
        name=MONO_MID_LOUDNORM_AUDIO_PROFILE,
        suffix=".m4a",
        filtergraph="pan=mono|c0=0.5*c0+0.5*c1,loudnorm=I=-16:TP=-1.5:LRA=11",
        channels=1,
        sample_rate=32000,
        codec="aac",
        bitrate="64k",
        movflags="+faststart",
    ),
    LEFT_ONLY_LOUDNORM_AUDIO_PROFILE: AudioTransformProfile(
        name=LEFT_ONLY_LOUDNORM_AUDIO_PROFILE,
        suffix=".m4a",
        filtergraph="pan=mono|c0=c0,loudnorm=I=-16:TP=-1.5:LRA=11",
        channels=1,
        sample_rate=32000,
        codec="aac",
        bitrate="64k",
        movflags="+faststart",
    ),
    RIGHT_ONLY_LOUDNORM_AUDIO_PROFILE: AudioTransformProfile(
        name=RIGHT_ONLY_LOUDNORM_AUDIO_PROFILE,
        suffix=".m4a",
        filtergraph="pan=mono|c0=c1,loudnorm=I=-16:TP=-1.5:LRA=11",
        channels=1,
        sample_rate=32000,
        codec="aac",
        bitrate="64k",
        movflags="+faststart",
    ),
    VOICE_BAND_MID_AUDIO_PROFILE: AudioTransformProfile(
        name=VOICE_BAND_MID_AUDIO_PROFILE,
        suffix=".m4a",
        filtergraph="pan=mono|c0=0.5*c0+0.5*c1,highpass=f=120,lowpass=f=3800,loudnorm=I=-16:TP=-1.5:LRA=11",
        channels=1,
        sample_rate=32000,
        codec="aac",
        bitrate="64k",
        movflags="+faststart",
    ),
}


def get_audio_profile(name: str) -> AudioTransformProfile:
    try:
        return AUDIO_TRANSFORM_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown audio profile: {name}") from exc


def list_audio_profiles(*names: str) -> tuple[AudioTransformProfile, ...]:
    if not names:
        return tuple(AUDIO_TRANSFORM_PROFILES.values())
    return tuple(get_audio_profile(name) for name in names)


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(FFMPEG_AUDIO_PROFILE_REQUIRED_MESSAGE)
    return ffmpeg


def render_audio_profile(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    profile: str | AudioTransformProfile,
    start_seconds: int | None = None,
    duration_seconds: int | None = None,
) -> Path:
    resolved_profile = (
        get_audio_profile(profile) if isinstance(profile, str) else profile
    )
    source = Path(source_path)
    destination = Path(destination_path)
    ffmpeg = _require_ffmpeg()

    command = [ffmpeg, "-hide_banner", "-y"]
    if start_seconds is not None:
        command.extend(["-ss", str(start_seconds)])
    command.extend(["-i", str(source), "-vn"])
    if duration_seconds is not None:
        command.extend(["-t", str(duration_seconds)])
    if resolved_profile.filtergraph is not None:
        command.extend(["-af", resolved_profile.filtergraph])
    if resolved_profile.channels is not None:
        command.extend(["-ac", str(resolved_profile.channels)])
    if resolved_profile.sample_rate is not None:
        command.extend(["-ar", str(resolved_profile.sample_rate)])
    if resolved_profile.codec is None:
        command.extend(["-c:a", "copy"])
    else:
        command.extend(["-c:a", resolved_profile.codec])
    if resolved_profile.bitrate is not None:
        command.extend(["-b:a", resolved_profile.bitrate])
    if resolved_profile.movflags is not None:
        command.extend(["-movflags", resolved_profile.movflags])
    command.append(str(destination))
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return destination
