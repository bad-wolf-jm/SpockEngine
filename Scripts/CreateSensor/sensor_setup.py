import os
from git import Repo, RootUpdateProgress

CMAKE_VERSION = '3.20'

SENSOR_NAME = None
SENSOR_VERSION = None

SENSOR_FOLDERS = [
    "Deps", "SensorAssets", "Scenes", "Source", "Generated", "Build", "Docs"
]


def generate_gitignore():
    with open('.gitignore', 'w') as out:
        out.write("Build/\n")
        out.write("Generated/\n")


def generate_readme():
    with open('README.md', 'w') as out:
        out.write("# New sensor")


def generate_cmakelists():
    with open('CMakeLists.txt', 'w') as out:
        out.write(f"cmake_minimum_required(VERSION {CMAKE_VERSION})\n")
        out.write(f"project({SENSOR_NAME} LANGUAGES CUDA CXX)\n")
        out.write(f"set(CMAKE_CXX_STANDARD 17)\n")
        out.write(f"\n")
        out.write(f"add_subdirectory(Deps/LTSimulationEngine)\n")


def generate_editor_start_script():
    with open('LaunchEditor.bat', 'w') as out:
        out.write(f"echo off\n")
        path = os.path.join(os.getcwd(), 'Deps',
                            'LTSimulationEngine', 'Build', 'Bin', "Editor.exe")
        out.write(f"{path}\n")


def generate_build_script():
    with open('BuildRuntime.bat', 'w') as out:
        out.write(f"echo off\n")
        out.write(f"cmake --build ./Build --target LTSimulationEngineRuntime\n")
        out.write(f"cmake --build ./Build --target Demo_2\n")


class SubmoduleUpdateProgressPrinter(RootUpdateProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        print(op_code, cur_count, max_count, cur_count /
              (max_count or 100.0), message or "NO MESSAGE")


if __name__ == "__main__":

    repo = Repo.init(os.getcwd())

    for folder_name in SENSOR_FOLDERS:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    ltse = repo.create_submodule("LTSimulationEngine", "Deps/LTSimulationEngine",
                                 url="git@svleddar-gitlab.leddartech.local:simulation/LTSimulationEngine.git", branch='development')
    repo.index.commit("Added LTSimulationEngine submodule")
    repo.submodule_update(
        recursive=True, progress=SubmoduleUpdateProgressPrinter())

    generate_gitignore()
    generate_readme()
    generate_cmakelists()
    generate_editor_start_script()
    generate_build_script()