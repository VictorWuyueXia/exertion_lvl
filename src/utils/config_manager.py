import yaml
import os
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """
    配置管理器
    用于加载、验证和管理YAML配置文件
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载WandB配置
        self._load_wandb_config()
        
        # 验证配置
        self._validate_config()
        
        return self.config
    
    def _load_wandb_config(self):
        """加载WandB配置文件"""
        wandb_config_path = "config/wandb_config.yaml"
        if os.path.exists(wandb_config_path):
            try:
                with open(wandb_config_path, 'r', encoding='utf-8') as f:
                    wandb_config = yaml.safe_load(f)
                
                # 合并WandB配置
                if 'wandb' not in self.config:
                    self.config['wandb'] = {}
                self.config['wandb'].update(wandb_config.get('wandb', {}))
                
                print("WandB配置已加载")
            except Exception as e:
                print(f"加载WandB配置失败: {e}")
        else:
            print("WandB配置文件不存在，使用默认配置")
            # 设置默认WandB配置
            self.config['wandb'] = {
                'enabled': True,
                'project': 'exertion-level-detection',
                'name': 'vgg16_wav2vec2_layer4_{timestamp}',
                'tags': ['vgg16', 'wav2vec2', 'exertion-detection'],
                'description': '基于VGG16和wav2vec2第4层的运动强度检测模型训练'
            }
    
    def _validate_config(self):
        """验证配置文件的完整性"""
        # 精简配置只需要基本节
        required_sections = ['data', 'training', 'system']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需的节: {section}")
        
        # 验证数据配置
        data_config = self.config['data']
        required_data_keys = ['feature_dir', 'label_file']
        for key in required_data_keys:
            if key not in data_config:
                raise ValueError(f"数据配置缺少必需的键: {key}")
        
        # 验证训练配置
        training_config = self.config['training']
        required_training_keys = ['epochs', 'batch_size', 'learning_rate']
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"训练配置缺少必需的键: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        config = self.config
        
        # 导航到父级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def save_config(self, output_path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def merge_config(self, other_config: Dict[str, Any]):
        """合并其他配置"""
        self._merge_dict(self.config, other_config)
    
    def _merge_dict(self, base: Dict[str, Any], update: Dict[str, Any]):
        """递归合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            else:
                base[key] = value
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return {
            'feature_dir': self.get('data.feature_dir'),
            'use_acoustic': self.get('data.features.use_acoustic', False),
            'use_mfcc': self.get('data.features.use_mfcc', True),
            'use_wav2vec2': self.get('data.features.use_wav2vec2', True),
            'wav2vec2_layers': self.get('data.features.wav2vec2_layers', [4]),
            'mfcc_dim': self.get('model.mfcc_dim', 40),  # 从配置文件读取
            'wav2vec2_dim': self.get('model.wav2vec2_dim', 768),  # 从配置文件读取
            'num_classes': self.get('model.num_classes', 5),  # 从配置文件读取
            'dropout_rate': self.get('model.dropout_rate', 0.5),  # 从配置文件读取
            'optimize_for_gpu': self.get('model.optimize_for_gpu', True),  # 从配置文件读取
            'batch_size': self.get('training.batch_size'),
            'epochs': self.get('training.epochs'),
            'learning_rate': self.get('training.learning_rate'),
            'weight_decay': self.get('training.weight_decay'),
            'optimizer': self.get('training.optimizer'),
            'scheduler': self.get('training.scheduler'),
            'criterion': self.get('training.criterion'),
            'use_amp': self.get('system.gpu.enable_amp', True),  # 从GPU配置读取
            'patience': self.get('training.patience'),
            'seed': self.get('system.seed'),
            'num_workers': self.get('training.num_workers'),
            'pin_memory': self.get('training.pin_memory'),  # 从训练配置读取
            'persistent_workers': self.get('training.persistent_workers'),  # 从训练配置读取
            'prefetch_factor': self.get('training.prefetch_factor'),  # 从训练配置读取
            'n_folds': self.get('training.n_folds'),
            # 添加数据平衡和分割配置
            'data': {
                'balance_data': self.get('data.balance_data', True),
                'balance_config': self.get('data.balance_config', {}),
                'split': self.get('data.split', {}),
                'normalize_data': self.get('data.normalize_data', False),
                'leakage_check': self.get('data.leakage_check', True),
            },
            'system': {
                'seed': self.get('system.seed', 42),
                'gpu': self.get('system.gpu', {}),
            },
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置（包含所有节）"""
        return self.config.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """获取完整配置（兼容性方法）"""
        return self.config.copy()
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'mfcc_dim': self.get('model.mfcc_dim'),  # 从配置文件读取
            'wav2vec2_dim': self.get('model.wav2vec2_dim'),  # 从配置文件读取
            'num_classes': self.get('model.num_classes'),  # 从配置文件读取
            'dropout_rate': self.get('model.dropout_rate'),  # 从配置文件读取
            'use_mfcc': self.get('data.features.use_mfcc', True),
            'use_wav2vec2': self.get('data.features.use_wav2vec2', True),
            'optimize_for_gpu': self.get('model.optimize_for_gpu', True),
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return {
            'metrics': self.get('evaluation.metrics'),
            'visualization': self.get('evaluation.visualization'),
            'threshold_optimization': self.get('evaluation.threshold_optimization'),
        }
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return {
            'seed': self.get('system.seed'),
            'gpu': self.get('system.gpu'),
            'multiprocessing': self.get('system.multiprocessing'),
        }
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("配置信息（精简版）")
        print("=" * 60)
        
        # 数据配置
        print(f"数据目录: {self.get('data.feature_dir')}")
        print(f"特征配置: MFCC={self.get('data.features.use_mfcc', True)}, wav2vec2={self.get('data.features.use_wav2vec2', True)}")
        print(f"wav2vec2层: {self.get('data.features.wav2vec2_layers', [4])}")
        
        # 训练配置
        print(f"训练轮数: {self.get('training.epochs')}")
        print(f"批大小: {self.get('training.batch_size')}")
        print(f"学习率: {self.get('training.learning_rate')}")
        print(f"优化器: {self.get('training.optimizer')}")
        print(f"调度器: {self.get('training.scheduler')}")
        print(f"交叉验证折数: {self.get('training.n_folds')}")
        
        # 系统配置
        print(f"随机种子: {self.get('system.seed')}")
        
        print("=" * 60)


def load_config_from_args() -> ConfigManager:
    """从命令行参数加载配置"""
    parser = argparse.ArgumentParser(description='运动强度检测模型训练')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--override', type=str, nargs='*',
                       help='覆盖配置项，格式: key=value')
    
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    
    # 处理覆盖参数
    if args.override:
        for override in args.override:
            if '=' in override:
                key, value = override.split('=', 1)
                # 尝试转换值类型
                try:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # 保持字符串类型
                
                config_manager.set(key, value)
    
    return config_manager


def create_experiment_config(experiment_name: str, **kwargs) -> ConfigManager:
    """创建实验配置"""
    # 加载默认配置
    config_manager = ConfigManager('config/training_config.yaml')
    
    # 设置实验名称
    config_manager.set('experiment.name', experiment_name)
    
    # 应用自定义参数
    for key, value in kwargs.items():
        config_manager.set(key, value)
    
    return config_manager
