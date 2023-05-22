Need to install paddlenlp.
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie#%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96




Here are some code running examples. You can also refer to the original website at: https://github.com/PaddlePaddle/PaddleNLP/tree/develop

1. 模型微调
推荐使用 Trainer API 对模型进行微调。只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 uie-base 作为预训练模型进行模型微调，将微调后的模型保存至$finetuned_model：

单卡启动：

export finetuned_model=./checkpoint/model_best

python finetune.py  \
    --device cpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 42 \
    --model_name_or_path uie-base \
    --output_dir $finetuned_model \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_length 512  \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size  16 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --label_names 'start_positions' 'end_positions' \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir $finetuned_model \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
    
如果在GPU环境中使用，可以指定gpus参数进行多卡训练：

export finetuned_model=./checkpoint/model_best

python -u -m paddle.distributed.launch --gpus "0,1" finetune.py \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 42 \
    --model_name_or_path uie-base \
    --output_dir $finetuned_model \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_length 512  \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size  16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir $finetuned_model \
    --label_names "start_positions" "end_positions" \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1 \


可配置参数说明：

model_name_or_path：必须，进行 few shot 训练使用的预训练模型。可选择的有 "uie-base"、 "uie-medium", "uie-mini", "uie-micro", "uie-nano", "uie-m-base", "uie-m-large"。
multilingual：是否是跨语言模型，用 "uie-m-base", "uie-m-large" 等模型进微调得到的模型也是多语言模型，需要设置为 True；默认为 False。
output_dir：必须，模型训练或压缩后保存的模型目录；默认为 None 。
device: 训练设备，可选择 'cpu'、'gpu' 、'npu'其中的一种；默认为 GPU 训练。
per_device_train_batch_size：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
per_device_eval_batch_size：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
learning_rate：训练最大学习率，UIE 推荐设置为 1e-5；默认值为3e-5。
num_train_epochs: 训练轮次，使用早停法时可以选择 100；默认为10。
logging_steps: 训练过程中日志打印的间隔 steps 数，默认100。
save_steps: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
seed：全局随机种子，默认为 42。
weight_decay：除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。可选；默认为 0.0；
do_train:是否进行微调训练，设置该参数表示进行微调训练，默认不设置。
do_eval:是否进行评估，设置该参数表示进行评估。
该示例代码中由于设置了参数 --do_eval，因此在训练完会自动进行评估。


2. 模型评估
通过运行以下命令进行模型评估：
    
python evaluate.py \
    --model_path ./checkpoint/model_best/checkpoint-200 \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512
    
通过运行以下命令对 UIE-M 进行模型评估：

python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512 \
    --multilingual
评估方式说明：采用单阶段评价的方式，即关系抽取、事件抽取等需要分阶段预测的任务对每一阶段的预测结果进行分别评价。验证/测试集默认会利用同一层级的所有标签来构造出全部负例。

可开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试：

python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --debug

可配置参数说明：

model_path: 进行评估的模型文件夹路径，路径下需包含模型权重文件model_state.pdparams及配置文件model_config.json。
test_path: 进行评估的测试集文件。
batch_size: 批处理大小，请结合机器情况进行调整，默认为16。
max_seq_len: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
debug: 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。
multilingual: 是否是跨语言模型，默认关闭。
schema_lang: 选择schema的语言，可选有ch和en。默认为ch，英文数据集请选择en。