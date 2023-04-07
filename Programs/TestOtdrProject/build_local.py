import os
from pathlib import Path
from subprocess import Popen

##
# VSTest.Console d:/OTDR/UnitTestOlm/bin/debug/UnitTestOlm.dll /TestCaseFilter:"TestCategory=Server"
##

program_files = Path("C:\\Program Files (x86)")
msbuild_path = program_files / "Microsoft Visual Studio" / "2019" / "Professional" / "MSBuild" / "Current" / "Bin"
msbuild_exe = msbuild_path / "MSBuild.exe"

processor_count = 16
configuration = "debug"
target = "AnyCPU"

git_repo_path = Path("D:\\OTDR")

otdr_base = git_repo_path / "OtdrBase" / "FF4"
otdr_module_names = ["Metrino.Otdr", "Metrino.Otdr.SignalProcessing", "Metrino.Otdr.Simulation", "Metrino.Otdr.FileConverter"]

otdr_instrument_base = git_repo_path / "OtdrInstrument" / "FF4"
otdr_instrument_module_names = ["Metrino.Otdr.Instrument"]

olm_base = git_repo_path / "OlmBase" / "FF4"
olm_module_names = ["Metrino.Olm", "Metrino.Olm.SignalProcessing"]

olm_instrument_base = git_repo_path / "OlmInstrument" / "FF4"
olm_instrument_module_names = ["Metrino.Olm.Instrument"]

olm_unit_test_base = git_repo_path
olm_unit_test_module_names = ["OlmTest"]

mono_base = Path("D:\\Build\\Source")
mono_module_names = ["Metrino.Interop"]

mono_tools_base = Path("D:\\Build\\Source\\Tools")
mono_tools_module_names = ["OlmDevTool"]

# console_logger_parameters = ['Summary', 'ErrorsOnly']
console_logger_parameters = ['Summary']

output_root = Path("C:\\GitLab\\SpockEngine\\Programs\\TestOtdrProject\\Lib")

def build_command_line(project_file, configuration, output_path, processor_count):
    clp = f"-clp:{';'.join(console_logger_parameters)}"
    output_path = f"/p:OutputPath={output_path}"
    configuration = f"/p:configuration={configuration}"
    thread_count = f"-m:{os.cpu_count()}"

    return [str(msbuild_exe), thread_count, clp, "-v:n", str(project_file), configuration, output_path]


if __name__ == '__main__':
    for module_name in otdr_module_names:
        project_path = otdr_base / module_name / f"{module_name}.csproj"
        module_output_path = output_root / configuration / module_name
        command_line = build_command_line(project_path, configuration, module_output_path, processor_count)
        print(command_line)
        process = Popen(command_line)
        process.communicate()

    for module_name in otdr_instrument_module_names:
        project_path = otdr_instrument_base / module_name / f"{module_name}.csproj"
        module_output_path = output_root / configuration / module_name
        command_line = build_command_line(project_path, configuration, module_output_path, processor_count)
        print(command_line)
        process = Popen(command_line)
        process.communicate()

    for module_name in olm_module_names:
        project_path = olm_base / module_name / f"{module_name}.csproj"
        module_output_path = output_root / configuration / module_name
        command_line = build_command_line(project_path, configuration, module_output_path, processor_count)
        print(command_line)
        process = Popen(command_line)
        process.communicate()

    for module_name in olm_instrument_module_names:
        project_path = olm_instrument_base / module_name / f"{module_name}.csproj"
        module_output_path = output_root / configuration / module_name
        command_line = build_command_line(project_path, configuration, module_output_path, processor_count)
        print(command_line)
        process = Popen(command_line)
        process.communicate()

    for module_name in mono_module_names:
        project_path = mono_base / module_name / f"{module_name}.csproj"
        module_output_path = output_root / configuration / module_name
        command_line = build_command_line(project_path, configuration, module_output_path, processor_count)
        print(command_line)
        process = Popen(command_line)
        process.communicate()
