mkdir -p log
mkdir -p log/mas
now=$(date +"%Y%m%d_%H%M%S")

root_dir=/mnt/lustre/$(whoami)
project_dir=$root_dir/easyfl/applications/mas
data_dir=$root_dir/datasets/taskonomy_datasets
client_file=$project_dir/clients.txt

export PYTHONPATH=$PYTHONPATH:${pwd}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p) partition="$2"; shift ;;
        -t) tasks="$2"; shift ;;
        -a) arch="$2"; shift ;;
        -e) local_epoch="$2"; shift ;;
        -k) clients_per_round="$2"; shift ;;
        -b) batch_size="$2"; shift ;;
        -r) rounds="$2"; shift ;;
        -lr) lr="$2"; shift ;;
        -lrt) lr_type="$2"; shift ;;
        -te) test_every="$2"; shift ;;
        -se) save_model_every="$2"; shift ;;
        -gpus) gpus="$2"; shift ;;
        -count) run_count="$2"; shift ;;
        -port) dist_port="$2"; shift ;;
        -tag) tag="$2"; shift ;;
        -tag_step) tag_step="$2"; shift ;;
        -what) what="$2"; shift ;;
        -client_id) client_id="$2"; shift ;;
        -agg_strategy) agg_strategy="$2"; shift ;;
        -pretrained) pretrained="$2"; shift ;;
        -pt) pretrained_tasks="$2"; shift ;;
        -decoder) decoder="$2"; shift ;;
        -half) half="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "${partition}" ]
  then
    partition=partition
fi

if [ -z "${tasks}" ]
  then
    tasks=""
fi

if [ -z "${arch}" ]
  then
    arch=xception # options: xception, resnet18
fi

if [ -z "${local_epoch}" ]
  then
    local_epoch=5
fi

if [ -z "${clients_per_round}" ]
  then
    clients_per_round=5
fi

if [ -z "${batch_size}" ]
  then
    batch_size=64
fi

if [ -z "${lr}" ]
  then
    lr=0.1
fi

if [ -z "${lr_type}" ]
  then
    lr_type=poly
fi

if [ -z "${rounds}" ]
  then
    rounds=100
fi

if [ -z "${test_every}" ]
  then
    test_every=1
fi

if [ -z "${save_model_every}" ]
  then
    save_model_every=1
fi

if [ -z "${gpus}" ]
  then
    gpus=1
fi

if [ -z "${dist_port}" ]
  then
    dist_port=23344
fi

# Whether use task affinity grouping (lookahead)
if [ -z "${tag}" ]
  then
    tag='y'
fi

# Lookahead step
if [ -z "${tag_step}" ]
  then
    tag_step=10
fi

if [ -z "${run_count}" ]
  then
    run_count=0
fi

if [ -z "${client_id}" ]
  then
    client_id='NA'
fi

if [ -z "${agg_strategy}" ]
  then
    agg_strategy='FedAvg'
fi

if [ -z "${pretrained_tasks}" ]
  then
    pretrained_tasks='sdnkt'
fi

use_pretrained='y'
if [ -z "${pretrained}" ]
  then
    pretrained='n'
    use_pretrained='n'
    pretrained_tasks='n'
fi

if [ -z "${decoder}" ]
  then
    decoder='y'
fi

if [ -z "${half}" ]
  then
    half='n'
fi

job_name=mas-${tasks}-${arch}-b${batch_size}-${lr_type}lr${lr}-${agg_strategy}-tag-${tag}-${tag_step}-e${local_epoch}-n${clients_per_round}-r${rounds}-te${test_every}-se${save_model_every}-pretrained-${use_pretrained}-${pretrained_tasks}-${what}-${run_count}
echo ${job_name}

srun -u --partition=${partition} --job-name=${job_name} \
    -n${gpus} --gres=gpu:${gpus} --ntasks-per-node=${gpus} \
    python ${project_dir}/main.py --data_dir ${data_dir} --arch ${arch} --client_file ${client_file} \
      --task_id ${job_name} --tasks ${tasks} --rotate_loss --batch_size ${batch_size} --lr ${lr} --lr_type ${lr_type} \
      --local_epoch ${local_epoch} --clients_per_round ${clients_per_round} --rounds ${rounds} \
      --test_every ${test_every} --save_model_every ${save_model_every} --random_selection --lookahead ${tag} --lookahead_step ${tag_step} \
      --dist_port ${dist_port} --run_count ${run_count} --load_decoder ${decoder} --half ${half} \
      --aggregation_strategy ${agg_strategy} --pretrained ${pretrained} --pretrained_tasks ${pretrained_tasks} \
      --client_id ${client_id} 2>&1 | tee log/mas/${job_name}.log &
