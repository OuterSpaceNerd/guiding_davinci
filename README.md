# Guiding_davinci
This code is modified from https://github.com/pohanchi/blackboxbot?fbclid=IwAR1ROB6h_Wr5uEMDuulq6XjMcVCTysaDDlNbhUUoh5fbcZd5ReJ9Avhy25c



## Update language model loss

### Train
<pre><code> python3 main.py \
--type length \
--mode \ finetune \
--model microsoft/DialoGPT-medium \
--discount_r 1.0 \
--agent example \
--log_interval 5 \
--sample_time 6 \
--seed 42 \
--max_pt_len 20 \
--inner_lr 5e-6 \
--exp_name [experiment name on wandb] \
--wandb online
</code></pre> 

* log_interval : steps to log wandb
* wandb : online for upload result to wandb, disabled for the opposite
* mode : required from original repo to make sample time correct
