from setuptools import find_packages, setup

package_name = "yoloe_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="snogcon",
    maintainer_email="user@example.com",
    description="ROS 2 package for YOLOE and depth estimation",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yoloe_depth = yoloe_perception.yoloe_depth_node:main",
            "yoloe_demo = yoloe_perception.yoloe_demo_node:main",
            "grasp_reasoning = yoloe_perception.grasp_reasoning_node:main",
        ],
    },
)
