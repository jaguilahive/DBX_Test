import os
import pathlib


def resolve_project_root_from_module():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_project_root_from_notebook():
    return os.path.abspath(os.path.join(os.getcwd(), ".."))


def get_sample_data_dir(project_root=None):
    root = project_root or resolve_project_root_from_module()
    return os.path.join(root, "sample_data")


def get_config_dir(project_root=None):
    root = project_root or resolve_project_root_from_module()
    return os.path.join(root, "configs")


def get_output_dir(project_root=None):
    root = project_root or resolve_project_root_from_module()
    return os.path.join(root, "output")


def get_eval_dir(project_root=None):
    root = project_root or resolve_project_root_from_module()
    return os.path.join(root, "eval")


def resolve_sample_file(relative_path, project_root=None):
    sample_dir = get_sample_data_dir(project_root)
    full_path = os.path.join(sample_dir, relative_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Sample file not found: {full_path}")
    return full_path


def resolve_config_file(filename, project_root=None):
    config_dir = get_config_dir(project_root)
    full_path = os.path.join(config_dir, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")
    return full_path


def list_sample_files(subdirectory=None, project_root=None):
    sample_dir = get_sample_data_dir(project_root)
    if subdirectory:
        sample_dir = os.path.join(sample_dir, subdirectory)
    if not os.path.isdir(sample_dir):
        return []
    return sorted([
        os.path.join(sample_dir, filename)
        for filename in os.listdir(sample_dir)
        if os.path.isfile(os.path.join(sample_dir, filename))
        and not filename.startswith(".")
    ])


def strip_file_scheme(workspace_uri):
    if workspace_uri.startswith("file:"):
        return workspace_uri[5:]
    return workspace_uri
