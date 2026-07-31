from importlib.resources import files


def test_package_resources_are_available() -> None:
    package_root = files("yolo26_app")

    assert package_root.joinpath("core/config_template.yaml").is_file()
    assert package_root.joinpath("ui/icons/nav-annotate.svg").is_file()


def test_gui_entry_point_is_importable() -> None:
    from yolo26_app.app import main

    assert callable(main)
