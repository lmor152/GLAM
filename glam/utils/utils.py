import os
import zipfile
from pathlib import Path

import numpy as np
import pkg_resources
import requests
from tqdm import tqdm


def choose(option1=lambda: None, option2=lambda: None, chance1=0.5):
    """
    randomly choose either option1 function or option 2

    Inputs:
        option1:           option 1 callable - default None
        option2:           option 2 callable - default None
        chance1:           probability for option 1 to be selected. Defaults to 50/50 odds
    Outputs:
        return:            output of randomly chosen function
    """
    if np.random.uniform() <= chance1:
        return option1()
    return option2()


def build_address(
    unit="",
    first_number="",
    first_number_suffix="",
    second_number="",
    street_name="",
    suburb="",
    town_city="",
    postcode="",
    human=True,
):
    head = ""
    if len(unit) > 0:
        head += unit
    if len(unit) > 0 and len(first_number) > 0:
        head += "/"
    head += first_number + first_number_suffix

    if len(second_number) > 0 and len(head) > 0:
        head += "-" + second_number
    else:
        head += second_number

    middle_parts = [street_name, suburb, town_city]
    middle_parts = [x for x in middle_parts if len(x) > 0]
    middle_parts = ", ".join(middle_parts)

    addy_parts = [head, middle_parts, postcode]
    addy_parts = [x for x in addy_parts if len(x) > 0]

    if human:
        addy = " ".join(addy_parts)
    else:
        addy = "|".join(addy_parts).replace(",", "")

    return addy


def check_package_dependency(package_name, desired_version=None):
    """
    desired_version should be of format '==1.0.0'
    """

    try:
        if desired_version:
            pkg_resources.require(f"{package_name}{desired_version}")
        else:
            pkg_resources.require(package_name)

    except pkg_resources.DistributionNotFound:
        if desired_version:
            raise RuntimeError(
                f"Missing optional dependency {package_name}{desired_version}"
            )
        raise RuntimeError(f"Missing optional dependency {package_name}")
    except pkg_resources.VersionConflict:
        print(
            f"{package_name} version {desired_version} is installed, but a different version is present. Version should be {desired_version}"
        )

    # if package in sys.modules:
    #     return None

    # found = sys.modules.get(package,None)
    # if found is None:
    #     if version is not None:
    #         raise ValueError(f"Missing optional dependency: {package}=={version}")
    #     else:
    #         raise ValueError(f"Missing optional dependency: {package}")

    # if version is not None:
    #     if version != found.__version__:
    #         raise ValueError(f"Missing optional dependency: {package}=={version}")

    # return None


def download_dependencies(deps_directory: str) -> None:
    url = "https://r2.lmor152.com/glam-deps.zip"
    deps_path = Path(deps_directory)
    zip_path = deps_path / "dependencies.zip"

    if not deps_path.exists():
        deps_path.mkdir()

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as file,
            tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(deps_directory)

    os.remove(zip_path)
