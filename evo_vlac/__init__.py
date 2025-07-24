# 项目元信息
__version__ = "1.0.0"
__author__ = "zhangqi"
__email__ = "zhangqi1@pjlab.org"
__description__ = "internEVO Critic and VLA"

# 导入主要组件
from . import utils

# 修复导入路径
from .utils import data_processing_vlm
from .utils import model_utils
from .utils.model_utils import GAC_model
from .utils import video_tool
# 包级别的便捷函数
def get_version():
    """获取项目版本"""
    return __version__

# 定义包的公开接口
__all__ = ["utils", "model_utils", "data_processing_vlm", "get_version","GAC_model","video_tool"]
