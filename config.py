import platform
import torch

class Config:
    app_title = 'vinux'
    show_os_label = "ОС: "
    show_ver_label = "Версия: "
    show_device_label = f"Устройство: "

    rain_width = 11
    rain_height = 2

    logo = '''\
██╗   ██╗██╗███╗   ██╗██╗   ██╗██╗  ██╗
██║   ██║██║████╗  ██║██║   ██║╚██╗██╔╝
██║   ██║██║██╔██╗ ██║██║   ██║ ╚███╔╝ 
╚██╗ ██╔╝██║██║╚██╗██║██║   ██║ ██╔██╗ 
 ╚████╔╝ ██║██║ ╚████║╚██████╔╝██╔╝ ██╗
  ╚═══╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝'''
    
    @classmethod
    def get_os_name(cls) -> str:
        try:
            info = platform.freedesktop_os_release()
            return info.get("NAME", "Linux")
        except AttributeError:
            return platform.system()
        
    @staticmethod
    def get_version():
        return "v1.0.0-alpha"
    
    @staticmethod
    def get_device():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device