"""Build and test a private local conda channel from one verified source archive."""

import argparse
import hashlib
import os
import subprocess
import tarfile
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Package the pinned companions and project without uploading anything."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path, help="prior-generator source archive to package")
    parser.add_argument("--conda", default=os.environ.get("CONDA_EXE", "conda"))
    parser.add_argument("--output-folder", type=Path, default=root / "dist" / "conda-channel")
    args = parser.parse_args()
    sdist = args.sdist.resolve(strict=True)
    output = args.output_folder.resolve()
    if output.exists():
        parser.error("--output-folder must not exist; do not mix artifacts from different builds")

    with tarfile.open(sdist) as archive:
        metadata_members = [
            member
            for member in archive.getmembers()
            if len(Path(member.name).parts) == 2 and Path(member.name).name == "PKG-INFO"
        ]
        if len(metadata_members) != 1:
            parser.error("source archive must contain one top-level PKG-INFO")
        if not metadata_members[0].isfile():
            parser.error("PKG-INFO must be a regular file")
        metadata_stream = archive.extractfile(metadata_members[0])
        if metadata_stream is None:
            parser.error("PKG-INFO must be a regular file")
        with metadata_stream:
            metadata = BytesParser().parse(metadata_stream)
    if metadata["Name"] != "prior-generator" or not metadata["Version"]:
        parser.error("source archive must identify the prior-generator distribution")
    with sdist.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    environment = {
        **os.environ,
        "PRIOR_GENERATOR_VERSION": metadata["Version"],
        "PRIOR_GENERATOR_SDIST": sdist.as_uri(),
        "PRIOR_GENERATOR_SDIST_SHA256": digest,
        "CONDA_CHANNEL_PRIORITY": "strict",
        "CONDA_SOLVER": "libmamba",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="conda-build-", dir=output.parent) as directory:
        workspace = Path(directory)
        channel = workspace / "channel"
        channel.mkdir()
        for index, name in enumerate(("pymc-extras", "pymc-marketing", "prior-generator")):
            channels = ["--channel", channel.as_uri()] if index else []
            subprocess.run(
                [
                    args.conda,
                    "build",
                    str(root / "conda" / "recipes" / name),
                    "--python",
                    "3.13",
                    "--package-format",
                    "conda",
                    "--no-anaconda-upload",
                    "--override-channels",
                    *channels,
                    "--channel",
                    "conda-forge",
                    "--croot",
                    str(workspace / "work"),
                    "--output-folder",
                    str(channel),
                ],
                env=environment,
                check=True,
            )
        channel.rename(output)
    print(f"Built and tested conda channel: {output}")


if __name__ == "__main__":
    main()
