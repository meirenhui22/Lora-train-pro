# LoRA train script by @Akegarasu modify by @bdsqlsz

#ѵ��ģʽ(Lora��db��sdxl_lora��Sdxl_db��sdxl_cn3l��stable_cascade_db��stable_cascade_lora��controlnet��hunyuan_lora��hunyuan_db��sd3_db��flux_lora��flux_db)
$train_mode = "flux_lora"
#����ѵ������---------------------------------------------------------------
#ģ������
$pretrained_model = "E:\lora-train\models\flux\flux1-dev-fp8.safetensors" # base model path | ��ģ·��
$clip_l = "E:\lora-train\models\flux\clip_l.safetensors"
$t5xxl = "H:\ComfyUI\models\clip\t5xxl_fp8_e4m3fn.safetensors"
$vae = "E:\lora-train\models\flux\ae.sft"

#���ݼ�ͼƬ����
$train_data_dir = "./train/houtu" # train dataset path | ѵ�����ݼ�·��
$reg_data_dir = ""	# reg dataset path | �������ݼ���·��
$training_comment = "a flux lora trained by xxx,the chufaci is xx" # training_comment | ѵ�����ܣ�����д����������ʹ�ô����ؼ���

#ѵ������
$resolution = "512,512" # image resolution w,h. ͼƬ�ֱ��ʣ���,�ߡ�֧�ַ������Σ��������� 64 ������
$batch_size = 1 # batch size һ����ѵ��ͼƬ�����������������Կ�������Ӧ���ߡ�
$max_train_epoches = 2 # max train epoches | ���ѵ�� epoch
$save_every_n_epochs = 1 # save every n epochs | ÿ N �� epoch ����һ��
$network_dim = 16 # network dim | ���� 4~128������Խ��Խ��
$network_alpha = 16 # network alpha | ������ network_dim ��ͬ��ֵ���߲��ý�С��ֵ���� network_dim��һ�� ��ֹ���硣Ĭ��ֵΪ 1��ʹ�ý�С�� alpha ��Ҫ����ѧϰ�ʡ�

# Learning rate | ѧϰ��
$lr = "1e-4"
$unet_lr = "5e-4"
$text_encoder_lr = "2e-5"
$lr_scheduler = "cosine_with_min_lr"
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch�Դ�6�ֶ�̬ѧϰ�ʺ���
# constant����������, constant_with_warmup �������Ӻ󱣳ֳ�������, linear �����������Լ���, polynomial �������Ӻ�ƽ��˥��, cosine ���Ҳ�����, cosine_with_restarts ���Ҳ�Ӳ������˲�����ֵ��
# ����cosine_with_min_lr(�ʺ�ѵ��lora)��warmup_stable_decay(�ʺ�ѵ��db)��inverse_sqrt
$lr_warmup_steps = 0 # warmup steps | ѧϰ��Ԥ�Ȳ�����lr_scheduler Ϊ constant �� adafactor ʱ��ֵ��Ҫ��Ϊ0������ lr_scheduler Ϊ constant_with_warmup ʱ��Ҫ��д���ֵ
$lr_decay_steps = 0.2 # decay steps | ѧϰ��˥������������ lr_scheduler Ϊwarmup_stable_decayʱ ��Ҫ��д��һ����10%�ܲ���
$lr_scheduler_num_cycles = 1 # restarts nums | �����˻��������������� lr_scheduler Ϊ cosine_with_restarts ʱ��Ҫ��д���ֵ
$lr_scheduler_timescale = 0 #times scale |ʱ�����ţ����� lr_scheduler Ϊ inverse_sqrt ʱ��Ҫ��д���ֵ��Ĭ��ͬlr_warmup_steps
$lr_scheduler_min_lr_ratio = 0.1 #min lr ratio |��Сѧϰ�ʱ��ʣ����� lr_scheduler Ϊ cosine_with_min_lr����warmup_stable_decay ʱ��Ҫ��д���ֵ��Ĭ��0

# Output settings | �������
$output_name = "flux-houtu-v1" # output model name | ģ�ͱ�������
$save_model_as = "safetensors" # model save ext | ģ�ͱ����ʽ ckpt, pt, safetensors
$mixed_precision = "bf16" # Ĭ��fp16,no,bf16��ѡ
$save_precision = "bf16" # Ĭ��fp16,fp32,bf16��ѡ
$full_fp16 = 0 #����ȫfp16ģʽ���Զ���Ͼ��ȱ�Ϊfp16������Լ�Դ�
$full_bf16 = 0 #ѡ��ȫbf16ѵ��������30ϵ�����Կ���
$fp8_base = 1 #����fp8ģʽ������Լ�Դ棬ʵ���Թ���
$fp8_base_unet = 0

#�������ͼƬ
$enable_sample = 1 #1������ͼ��0����
$sample_at_first = 1 #�Ƿ���ѵ����ʼʱ�ͳ�ͼ
$sample_every_n_epochs = 1 #ÿn��epoch��һ��ͼ
$sample_prompts = "./prompts/flux-prompts.txt" #prompt�ļ�·��
$sample_sampler = "euler_a" #������ 'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'
$seed = 1026 # reproducable seed | �����ܲ����õ����ӣ�����һ��prompt��������Ӵ���ʵõ�ѵ��ͼ�����������Դ����ؼ���
#����ѵ������END---------------------------------------------------------------


# Train data path | ����ѵ����ģ�͡�ͼƬ
$is_v2_model = 0 # SD2.0 model | SD2.0ģ�� 2.0ģ���� clip_skip Ĭ����Ч
$v_parameterization = 1 # parameterization | ������ v2 ��512�����ֱ��ʰ汾����ʹ�á�
$network_weights = "" # pretrained weights for LoRA network | ����Ҫ�����е� LoRA ģ���ϼ���ѵ��������д LoRA ģ��·����
$network_multiplier = 1.0 # loraȨ�ر�����Ĭ��1.0
$dataset_class = ""
#$dataset_config = "./toml/datasets_qinglong.toml" # dataset config | ���ݼ������ļ�·��
$disable_mmap_load_safetensors = 0 #��wsl�¼���ģ���ٶ�����

#stable_cascade ѵ����ز���
$effnet_checkpoint_path = "./VAE/effnet_encoder.safetensors" #effnet���൱����������VAE
$stage_c_checkpoint_path = "./Stable-diffusion/train/stage_c_bf16.safetensors" #stage_c���൱��base_model
$text_model_checkpoint_path = "" #te�ı�����������һ��Ĭ�ϲ��������Զ���HF����
$save_text_model = 1 #0�ر�1��������һ��ѵ�����ñ���TE��λ�ã�֮����Ҫʹ�ã�ֻ��Ҫͨ��ǰ��Ĳ���text_model_checkpoint_path��ȡ����
$previewer_checkpoint_path = "./Stable-diffusion/train/previewer.safetensors" #Ԥ��ģ�ͣ�����Ԥ��ͼ�Ļ���Ҫʹ�á�
$adaptive_loss_weight = 1 #0�ر�1������ʹ��adaptive_loss_weight���ٷ��Ƽ����ر���ʹ��P2LOSSWIGHT

#SD3 ѵ����ز���
$clip_g = "./clip/clip_g.safetensors"
$t5xxl_device = "" #Ĭ��cuda���Դ治���ɸ�ΪCPU�����Ǻ���
$t5xxl_dtype = "bf16" #Ŀǰ֧��fp32��fp16��bf16
$text_encoder_batch_size = 12
$num_last_block_to_freeze = 0
$discrete_flow_shift = 3.185 # Euler ��ɢ����������ɢ��λ�ƣ�sd3Ĭ��Ϊ3.0
$apply_t5_attn_mask = 1 # �Ƿ�Ӧ��T5��ע�������룬Ĭ��Ϊ0

#flux ��ز���
$ae = $vae
$timestep_sampling = "sigmoid" # ʱ�䲽������������ѡ sd3��"sigma"����ͨDDPM��"uniform" �� flux��"sigmoid" ���� "shift". shift��Ҫ�޸�discarete_flow_shift�Ĳ���
$sigmoid_scale = 1.0 # sigmoid �������������ӣ�Ĭ��Ϊ 1.0���ϴ��ֵ��ʹ�������Ӿ���
$model_prediction_type = "raw" # ģ��Ԥ�����ͣ���ѡ flux��"raw"��������������"additive" �� sdѡ"sigma_scaled"
$guidance_scale = 1.0 # guidance scale������CFG, Ĭ��Ϊ 1.0
$blockwise_fused_optimizers = 1 # �Ƿ�ʹ�ÿ鼶�ں��Ż�����Ĭ��Ϊ1
$double_blocks_to_swap = 6 # �����Ŀ�����Ĭ��Ϊ6
$single_blocks_to_swap = 0 # �����Ŀ�����Ĭ��Ϊ0
$cpu_offload_checkpointing = 0 # �Ƿ�ʹ��CPUж��checkpoint��finetuneĬ�Ͽ���
$mem_eff_save = 1 # �Ƿ�ʹ���ڴ��Ч���棬Ĭ��Ϊ1
$split_qkv = 0 # �Ƿ����QKV��Ĭ��Ϊ1
$train_t5xxl = 0 #ѵ��T5
$split_mode = 0 # �Ƿ����ģʽ��Ĭ��Ϊ0, ������ֻѵ����������Դ�ӿ�ѵ���ٶȡ�

#����������
$base_weights = "" #ָ���ϲ�����ģbasemodel�е�ģ��·��������ÿո������Ĭ��Ϊ�գ���ʹ�á�
$base_weights_multiplier = "1.0" #ָ���ϲ�ģ�͵�Ȩ�أ�����ÿո������Ĭ��Ϊ1.0��

$gradient_checkpointing = 1 #�ݶȼ�飬������ɽ�Լ�Դ棬�����ٶȱ���
$gradient_accumulation_steps = 1 # �ݶ��ۼ�����������Ŵ�batchsize�ı���
$optimizer_accumulation_steps = 0

$train_unet_only = 1 # train U-Net only | ��ѵ�� U-Net���������������Ч����������Դ�ʹ�á�6G�Դ���Կ���
$train_text_encoder_only = 0 # train Text Encoder only | ��ѵ�� �ı�������

#LORA_PLUS
$enable_lora_plus = 0
$loraplus_lr_ratio = 16
$loraplus_unet_lr_ratio = 16
$loraplus_text_encoder_lr_ratio = 4

#dropout | �׳�(Ŀǰ��lycoris�����ݣ���ʹ��lycoris�Դ�dropout)
$network_dropout = 0 # dropout �ǻ���ѧϰ�з�ֹ���������ϵļ���������0.1~0.3 
$scale_weight_norms = 1.0 #��� dropout ʹ�ã������Լ�����Ƽ�1.0
$rank_dropout = 0 #loraģ�Ͷ�����rank�����dropout���Ƽ�0.1~0.3��δ���Թ���
$module_dropout = 0 #loraģ�Ͷ�����module�����dropout(���Ƿֲ�ģ���)���Ƽ�0.1~0.3��δ���Թ���
$caption_dropout_every_n_epochs = 0 #dropout caption
$caption_dropout_rate = 0 #0~1
$caption_tag_dropout_rate = 0 #0~1

#noise | ����
$noise_offset = 0 # help allow SD to gen better blacks and whites��(0-1) | ����SD���÷ֱ�ڰף��Ƽ�����0.06������0.1
$adaptive_noise_scale = 0 #����Ӧƫ�Ƶ�����10%~100%��noiseoffset��С
$noise_offset_random_strength = 0 #�������ǿ��
$multires_noise_iterations = 0 #��ֱ���������ɢ�������Ƽ�6-10,0���á�
$multires_noise_discount = 0 #��ֱ����������ű������Ƽ�0.1-0.3,����ص��Ļ����á�
$min_snr_gamma = 0 #��С�����٤��ֵ�����ٵ�stepʱlossֵ����ѧϰЧ�����á��Ƽ�3-5��5��ԭģ�ͼ���û��̫��Ӱ�죬3��ı����ս�����޸�Ϊ0���á�
$ip_noise_gamma = 0 #���������ӣ���ֹ����ۼ�
$ip_noise_gamma_random_strength = 0 #����������ǿ��
$debiased_estimation_loss = 0 #���������������minsnr�߼���
$loss_type = "l2" #��ʧ�������ͣ�`smooth_l1`��`huber`��`l2`(����MSE)
$huber_schedule = "snr" #huber����������ѡ `exponential`��`constant` �� `snr`
$huber_c = 0.1 #huber��ʧ������c����
$immiscible_noise = 0 #�Ƿ����������

#optimizer | �Ż���
$optimizer_type = "PagedAdamW8bit"
# ��ѡ�Ż���"adaFactor","AdamW","AdamW8bit","Lion","SGDNesterov","SGDNesterov8bit","DAdaptation",  
# �����Ż���"Lion8bit"(�ٶȸ��죬�ڴ����ĸ���)��"DAdaptAdaGrad"��"DAdaptAdan"(���������㷨��Ч������)��"DAdaptSGD"
# ����DAdaptAdam��DAdaptLion��DAdaptAdanIP��ǿ���Ƽ�DAdaptAdam
# �����Ż���"Sophia"(2����1.7���Դ�)��"Prodigy"����Ż�����������ӦDylora
# PagedAdamW8bit��PagedLion8bit��Adan��Tiger
# AdamWScheduleFree��SGDScheduleFree
# StableAdamW��Ranger
# came
$d_coef = "0.5" #prodigy D�����ٶ�
$d0 = "1e-4" #dadaptation�Լ�prodigy��ʼѧϰ��
$fused_backward_pass = 0 #ѵ����ģ��float32����ר�ý�Լ�Դ棬�����Ż���adafactor����adamw��gradient_accumulation_steps����Ϊ1���߲�����
$fused_optimizer_groups = 0

#gorkfast | �������
$gradfilter_ema_alpha = 0 #EMA�Ķ��������� ����ema_alpha������gradfilter_ema���Ƽ�0.98��Ϊ0��ر�
$gradfilter_ema_lamb = 2.0 #�˲���ema�ķŴ����ӳ�������

# ���ݼ����� ���captain���
$shuffle_caption = 1 # �������tokens
$keep_tokens = 1 # keep heading N tokens when shuffling caption tokens | ��������� tokens ʱ������ǰ N �����䡣
$prior_loss_weight = 1 #����Ȩ��,0-1
$weighted_captions = 0 #Ȩ�ش�꣬Ĭ��ʶ���ǩȨ�أ��﷨ͬwebui�����÷�������(abc), [abc],(abc:1.23),���ǲ����������ڼӶ��ţ������޷�ʶ��һ���ļ����75��tokens��
$secondary_separator = ";;;" #��Ҫ�ָ��������÷ָ����ָ��Ĳ��ֽ�����Ϊһ��token������ϴ�ƺͶ�����Ȼ���� caption_separator ȡ�������磬���ָ�� aaa;;bbb;;cc�������� aaa,bbb,cc ȡ����һ������
$keep_tokens_separator = "|||" #�����������䣬�������
$enable_wildcard = 0 #ͨ�������鿨����ʽ�ο� {aaa|bbb|ccc} 
$caption_prefix = "" #���ǰ׺�����Լ��������������ģ��Ҫ������masterpiece, best quality,
$caption_suffix = "" #����׺�����Լ��������ͷ�����Ҫ������full body��
$alpha_mask = 0 #�Ƿ�ʹ��͸���ɰ���

# Resume training state | �ָ�ѵ������
$save_state = 0 # save training state | ����ѵ��״̬ ���������� <output_name>-??????-state ?????? ��ʾ epoch ��
$resume = "" # resume from state | ��ĳ��״̬�ļ����лָ�ѵ�� ������Ϸ�����ͬʱʹ�� ���ڹ淶�ļ����� epoch ����ȫ�ֲ������ᱣ�� ��ʹ�ָ�ʱ����Ҳ�� 1 ��ʼ �� network_weights �ľ���ʵ�ֲ�������һ��
$save_state_on_train_end = 0 #ֻ��ѵ��������󱣴�ѵ��״̬

#����toml�ļ�
$output_config = 0 #������ֱ�����һ��toml�����ļ��������޷�ͬʱѵ������Ҫ�رղ�������ѵ����
$config_file = "./toml/" + $output_name + ".toml" #����ļ�����Ŀ¼���ļ����ƣ�Ĭ����ģ�ͱ���ͬ����

#wandb ��־ͬ��
$wandb_api_key = "" # wandbAPI KEY�����ڵ�¼

# ��������
$enable_bucket = 1 #������Ͱ
$min_bucket_reso = 256 # arb min resolution | arb ��С�ֱ���
$max_bucket_reso = 2048 # arb max resolution | arb ���ֱ���
$bucket_no_upscale = 1 #��Ͱ���Ŵ�
$persistent_workers = 1 # makes workers persistent, further reduces/eliminates the lag in between epochs. however it may increase memory usage | �ܵĸ��죬���ڴ档���������2��
$vae_batch_size = 4 #vae�������С��2-4
$clip_skip = 2 # clip skip | ��ѧ һ���� 2
$cache_latents = 1 #����Ǳ����
$cache_latents_to_disk = 1 # ����ͼƬ���̣��´�ѵ������Ҫ���»��棬1����0����
$torch_compile = 0 #ʹ��torch���빦�ܣ���Ҫ�汾����2.1
$dynamo_backend = "inductor" #"eager", "aot_eager", "inductor","aot_ts_nvfuser","nvprims_nvfuser","cudagraphs","aot_torchxla_trace_once"����ѵ��
$TORCHINDUCTOR_FX_GRAPH_CACHE = 1 #���ñ��� FX ͼ���档
$TORCHINDUCTOR_CACHE_DIR = "./torch_compile_cache" #ָ�����д��̻����λ�á�

#lycoris���
$enable_lycoris = 0 # ����lycoris
$conv_dim = 0 #��� dim���Ƽ���32
$conv_alpha = 0 #��� alpha���Ƽ�1����0.3
$algo = "lokr" # algo������ָ��ѵ��lycorisģ�����࣬
#����lora(����locon)��
#loha
#IA3
#lokr
#dylora
#full(DreamBooth��ѵ��Ȼ�󵼳�lora)
#diag-oft
#��ͨ��ѵ�������ڸ�������������任������������������
#����ԭʼ���ģ����������ٶȱ� LoRA ���죬���������ʵ�顣
#dim �������С���Ӧ������������̶��������С������������������ʹ���� LoRA ���߿ɱ��ԡ�

$dropout = 0 #lycorisר��dropout
$preset = "attn-mlp" #Ԥ��ѵ��ģ������
#full: default preset, train all the layers in the UNet and CLIP|Ĭ�����ã�ѵ������Unet��Clip��
#full-lin: full but skip convolutional layers|���������
#attn-mlp: train all the transformer block.|kohya���ã�ѵ������transformerģ��
#attn-only��only attention layer will be trained, lot of papers only do training on attn layer.|ֻ��ע������ᱻѵ�����ܶ�����ֻ��ע���������ѵ����
#unet-transformer-only�� as same as kohya_ss/sd_scripts with disabled TE, or, attn-mlp preset with train_unet_only enabled.|��attn-mlp���ƣ����ǹر�teѵ��
#unet-convblock-only�� only ResBlock, UpSample, DownSample will be trained.|ֻѵ�����ģ�飬����res�����²���ģ��
#./toml/example_lycoris.toml: Ҳ����ֱ��ʹ�����������ļ����ƶ��������ģ��ʹ�ò�ͬ�㷨ѵ������Ҫ����λ���ļ�·�����ο���������ӡ�

$factor = 8 #ֻ������lokr�����ӣ�-1~8��8Ϊȫά��
$decompose_both = 0 #������lokr�Ĳ������� LoKr �ֽ��������������ִ�� LoRA �ֽ⣨Ĭ�������ֻ�ֽ�ϴ�ľ���
$block_size = 4 #������dylora,�ָ������λ����С1Ҳ������һ��4��8��12��16�⼸��ѡ
$use_tucker = 0 #�����ڳ� (IA)^3 ��full
$use_scalar = 0 #���ݲ�ͬ�㷨���Զ�������ʼȨ��
$train_norm = 0 #��һ����
$dora_wd = 1 #Dora�����ֽ⣬��rankʹ�á�������LoRA, LoHa, ��LoKr
$full_matrix = 0  #ȫ����ֽ�
$bypass_mode = 0 #ͨ��ģʽ��רΪ bnb 8 λ/4 λ���Բ���ơ�(QLyCORIS)������LoRA, LoHa, ��LoKr
$rescaled = 1 #�������������ţ�Ч����ͬ��OFT
$constrain = 0 #����ֵΪFLOAT��Ч����ͬ��COFT

#dylora���
$enable_dylora = 0 # ����dylora����lycoris��ͻ��ֻ�ܿ�һ����
$unit = 4 #�ָ������λ����С1Ҳ������һ��4��8��12��16�⼸��ѡ

#Lora_FA
$enable_lora_fa = 0 # ����lora_fa����lycoris��dylora��ͻ��ֻ�ܿ�һ����

#oft
$enable_oft = 0 # ����oft�������ϳ�ͻ��ֻ�ܿ�һ����

# block weights | �ֲ�ѵ��
$enable_block_weights = 0 #�����ֲ�ѵ������lycoris��ͻ��ֻ�ܿ�һ����
$down_lr_weight = "1,0.2,1,1,0.2,1,1,0.2,1,1,1,1" #12�㣬��Ҫ��д12�����֣�0-1.Ҳ����ʹ�ú���д����֧��sine, cosine, linear, reverse_linear, zeros���ο�д��down_lr_weight=cosine+.25 
$mid_lr_weight = "1"  #1�㣬��Ҫ��д1�����֣�����ͬ�ϡ�
$up_lr_weight = "1,1,1,1,1,1,1,1,1,1,1,1"   #12�㣬ͬ���ϡ�
$block_lr_zero_threshold = 0  #����ֲ�Ȩ�ز��������ֵ����ôֱ�Ӳ�ѵ����Ĭ��0��

$enable_block_dim = 0 #����dim�ֲ�ѵ��
$block_dims = "128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128" #dim�ֲ㣬25��
$block_alphas = "16,16,32,16,32,32,64,16,16,64,64,64,16,64,16,64,32,16,16,64,16,16,16,64,16"  #alpha�ֲ㣬25��
$conv_block_dims = "32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32" #convdim�ֲ㣬25��
$conv_block_alphas = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" #convalpha�ֲ㣬25��

# block lr
$enable_block_lr = 0
$block_lr = "0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,0,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,$lr,0"

#SDXLר�ò���
#https://www.bilibili.com/video/BV1tk4y137fo/
$min_timestep = 0 #��Сʱ��Ĭ��ֵ0
$max_timestep = 1000 #���ʱ��Ĭ��ֵ1000
$cache_text_encoder_outputs = 1 #���������ı�������������������Դ�ʹ�á������޷���shuffle����
$cache_text_encoder_outputs_to_disk = 1 #���������ı�������������������Դ�ʹ�á������޷���shuffle����
$no_half_vae = 0 #��ֹ�뾫�ȣ���ֹ��ͼ���޷���mixed_precision��Ͼ��ȹ��á�
$bucket_reso_steps = 32 #SDXL��Ͱ����ѡ��32����64��32����ϸ��Ͱ��Ĭ��Ϊ64

#db checkpoint train
$stop_text_encoder_training = 0
$no_token_padding = 0 #�����зִ������

#sdxl_db
$diffusers_xformers = 0
$train_text_encoder = 0
$learning_rate_te1 = "5e-8"
$learning_rate_te2 = "5e-8"

#sdxl_cn3l or controlnet
$controlnet_model_name_or_path = ""
$conditioning_data_dir = ""
$cond_emb_dim = 32
$masked_loss = 0 #�����ɰ�loss��������ͼ����Rͨ��255��Ϊ����mask��0��Ϊ������

#�࿨����
$multi_gpu = 0                         #multi gpu | ���Կ�ѵ�����أ�0��1���� �ò����������Կ��� >= 2 ʹ��
$deepspeed = 0                         #deepspeed | ʹ��deepspeedѵ����0��1���� �ò����������Կ��� >= 2 ʹ��
$zero_stage = 2                        #zero stage | zero stage 0,1,2,3,�׶�2����ѵ�� �ò����������Կ��� >= 2 ʹ��
$offload_optimizer_device = ""      #offload optimizer device | �Ż��������豸��cpu����nvme, �ò����������Կ��� >= 2 ʹ��
$fp16_master_weights_and_gradients = 0 #fp16 master weights and gradients | fp16��Ȩ�غ��ݶȣ�0��1���� �ò����������Կ��� >= 2 ʹ��

$ddp_timeout = 120
$ddp_gradient_as_bucket_view = 1
$ddp_static_graph = 1

# ============= DO NOT MODIFY CONTENTS BELOW | �����޸��·����� =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$network_module = "networks.lora"
$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()
$laungh_script = "train_network"

if ($train_mode -ieq "stable_cascade_lora") {
  $laungh_script = "stable_cascade_train_c_network"
}
elseif ($train_mode -ieq "flux_lora") {
  $network_module = $network_module + "_flux"
  if ($split_mode -ne 0 -or $split_qkv -ne 0 -or $train_t5xxl -ne 0) {
    [void]$ext_args.Add("--network_args")
    if ($split_qkv -ne 0 -and $enable_lycoris -ne 1) {
      [void]$ext_args.Add("split_qkv=True")
    }
    if ($train_t5xxl -ne 0) {
      $cache_text_encoder_outputs = 0
      $cache_text_encoder_outputs_to_disk = 0
      [void]$ext_args.Add("train_t5xxl=True")
    }
    if ($split_mode -ne 0 -and $enable_lycoris -ne 1) {
      [void]$ext_args.Add("train_blocks=single")
      [void]$ext_args.Add("--split_mode")
    }
  }
}

if ($multi_gpu -eq 1) {
  $launch_args += "--multi_gpu"
  $launch_args += "--mixed_precision=$mixed_precision"
  $blockwise_fused_optimizers = 0
  if ($deepspeed -eq 1) {
    [void]$ext_args.Add("--deepspeed")
    if ($zero_stage -ne 0) {
      [void]$ext_args.Add("--zero_stage=$zero_stage")
    }
    if ($offload_optimizer_device) {
      [void]$ext_args.Add("--offload_optimizer_device=$offload_optimizer_device")
    }
    if ($fp16_master_weights_and_gradients -eq 1) {
      [void]$ext_args.Add("--fp16_master_weights_and_gradients")
    }
  }
  if ($ddp_timeout -ne 0) {
    [void]$ext_args.Add("--ddp_timeout=$ddp_timeout")
  }
  if ($ddp_gradient_as_bucket_view -ne 0) {
    [void]$ext_args.Add("--ddp_gradient_as_bucket_view")
  }
  if ($ddp_static_graph -ne 0) {
    [void]$ext_args.Add("--ddp_static_graph")
  }
}

if (-not ($train_mode -ilike "*lora")) {
  $network_module = ""
  $network_alpha = ""
  $conv_dim = ""
  $conv_alpha = ""
  $network_weights = ""
  $network_multiplier = 1.0
  $enable_block_weights = 0
  $enable_block_dim = 0
  $enable_lycoris = 0
  $enable_dylora = 0
  $enable_lora_fa = 0
  $enable_oft = 0
  $unet_lr = ""
  $text_encoder_lr = ""
  $train_unet_only = 0
  $train_text_encoder_only = 0
  $training_comment = ""
  $prior_loss_weight = 1
  $network_dropout = "0"
  $enable_lora_plus = 0
  if ($train_mode -ine "sdxl_cn3l") {
    $network_dim = ""
  }
}

if ($train_mode -ilike "*db") {
  if ($train_mode -ieq "db") {
    $laungh_script = "train_db";
    if ($no_token_padding -ne 0) {
      [void]$ext_args.Add("--no_token_padding")
    }
    if ($stop_text_encoder_training) {
      if ($gradient_accumulation_steps) {
        $stop_text_encoder_training = $stop_text_encoder_training * $gradient_accumulation_steps
      }
      [void]$ext_args.Add("--stop_text_encoder_training=$stop_text_encoder_training")
    }
    if ($learning_rate_te) {
      [void]$ext_args.Add("--learning_rate_te=$learning_rate_te")
    }
  }
  else {
    if ($train_mode -ieq "stable_cascade_db") {
      $laungh_script = "stable_cascade_train_stage_c"
      $pretrained_model = ""
      $learning_rate_te2 = 0
      $min_snr_gamma = 0
      $ip_noise_gamma = 0
      $loss_type = "l2"
      $weighted_captions = 0
      $debiased_estimation_loss = 0
      $immiscible_noise = 0
    }
    else {
      $laungh_script = "train"
      if ($train_mode -ieq "sdxl_db") {
        if ($diffusers_xformers -ne 0) {
          [void]$ext_args.Add("--diffusers_xformers")
        }
        if ($train_text_encoder -ne 0) {
          [void]$ext_args.Add("--train_text_encoder")
          if ($learning_rate_te1 -ne 0) {
            [void]$ext_args.Add("--learning_rate_te1=$learning_rate_te1")
          }
          if ($learning_rate_te2 -ne 0) {
            [void]$ext_args.Add("--learning_rate_te2=$learning_rate_te2")
          }
        }
        if ($enable_block_lr -ne 0) {
          [void]$ext_args.Add("--block_lr=$block_lr")   
        }
      }
      elseif ($train_mode -ieq "flux_db") {
        $mem_eff_save = 1
        if ($blockwise_fused_optimizers -ne 0) {
          [void]$ext_args.Add("--blockwise_fused_optimizers")
        }
        if ($double_blocks_to_swap -ne 0) {
          [void]$ext_args.Add("--double_blocks_to_swap=$double_blocks_to_swap")
        }
        if ($single_blocks_to_swap -ne 0) {
          [void]$ext_args.Add("--single_blocks_to_swap=$single_blocks_to_swap")
        }
        if ($cpu_offload_checkpointing -ne 0) {
          [void]$ext_args.Add("--cpu_offload_checkpointing")
        }
        if ($mem_eff_save -ne 0) {
          [void]$ext_args.Add("--mem_eff_save")
        }
      }
    }
  }
}

if ($train_mode -ilike "*cn3l" -or $train_mode -ilike "*controlnet") {
  if ($train_mode -ilike "*controlnet") {
    $laungh_script = "train_controlnet"
    if ($controlnet_model_name_or_path) {
      [void]$ext_args.Add("--controlnet_model_name_or_path=$controlnet_model_name_or_path")
    }
  }
  else {
    $laungh_script = "train_control_net_lllite"
    if ($cond_emb_dim) { 
      [void]$ext_args.Add("--cond_emb_dim=$cond_emb_dim")
    }
  }
  if ($conditioning_data_dir) { 
    [void]$ext_args.Add("--conditioning_data_dir=$conditioning_data_dir")
  }
  if ($masked_loss) { 
    [void]$ext_args.Add("--masked_loss")
  }
}

if ($train_mode -ilike "sdxl*") {
  $laungh_script = "sdxl_" + $laungh_script
  if ($min_timestep -ne 0) {
    [void]$ext_args.Add("--min_timestep=$min_timestep")
  }
  if ($max_timestep -ne 1000) {
    [void]$ext_args.Add("--max_timestep=$max_timestep")
  }
  if ($cache_text_encoder_outputs -ne 0) { 
    [void]$ext_args.Add("--cache_text_encoder_outputs")
    if ($cache_text_encoder_outputs_to_disk -ne 0) { 
      [void]$ext_args.Add("--cache_text_encoder_outputs_to_disk")
    }
    $shuffle_caption = 0
    $loraplus_text_encoder_lr_ratio = 0
    $caption_dropout_rate = 0
    $caption_tag_dropout_rate = 0
  }
  if ($no_half_vae -ne 0) { 
    [void]$ext_args.Add("--no_half_vae")
    $mixed_precision = ""
    $full_fp16 = 0
    $full_bf16 = 0
    $fp8_base = 0
    $fp8_base_unet = 0
  }
  if ($bucket_reso_steps -ne 64) { 
    [void]$ext_args.Add("--bucket_reso_steps=$bucket_reso_steps")
  }
}

if ($train_mode -ilike "sd3*" -or $train_mode -ilike "flux*") {
  if ($clip_l) {
    [void]$ext_args.Add("--clip_l=$clip_l")
  }
  if ($t5xxl) {
    [void]$ext_args.Add("--t5xxl=$t5xxl")
  }
  if ($apply_t5_attn_mask) {
    [void]$ext_args.Add("--apply_t5_attn_mask")
  }
  if ($num_last_block_to_freeze -ne 0) {
    [void]$ext_args.Add("--num_last_block_to_freeze=$num_last_block_to_freeze")
  }
  if ($discrete_flow_shift) {
    [void]$ext_args.Add("--discrete_flow_shift=$discrete_flow_shift")
  }
  if ($train_mode -ilike "flux*") {
    $laungh_script = "flux_" + $laungh_script
    $enable_dylora = 0
    $enable_lora_fa = 0
    $enable_oft = 0
    $enable_block_lr = 0
    $enable_block_weights = 0
    $enable_block_dim = 0
    if ($timestep_sampling) {
      [void]$ext_args.Add("--timestep_sampling=$timestep_sampling")
    }
    if ($sigmoid_scale) {
      [void]$ext_args.Add("--sigmoid_scale=$sigmoid_scale")
    }
    if ($model_prediction_type) {
      [void]$ext_args.Add("--model_prediction_type=$model_prediction_type")
    }
    if ($guidance_scale) {
      [void]$ext_args.Add("--guidance_scale=$guidance_scale")
    }
    if ($guidance_rescale -ne 0) {
      [void]$ext_args.Add("--guidance_rescale")
    }
    if ($ae) {
      [void]$ext_args.Add("--ae=$ae")
      $vae = ""
    }
    if ($torch_compile) {
      # $Env:CC = "E:\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64\cl.exe"
      [void]$ext_args.Add("--torch_compile")
      if ($dynamo_backend) {
        [void]$ext_args.Add("--dynamo_backend=$dynamo_backend")
      }
      if ($TORCHINDUCTOR_FX_GRAPH_CACHE -ne 0) {
        $Env:TORCHINDUCTOR_FX_GRAPH_CACHE = "1"
        if ($TORCHINDUCTOR_CACHE_DIR) {
          $Env:TORCHINDUCTOR_CACHE_DIR = $TORCHINDUCTOR_CACHE_DIR
        }
      }
    }
  }
  else {
    $laungh_script = "sd3_" + $laungh_script
    if ($clip_g) {
      [void]$ext_args.Add("--clip_g=$clip_g")
    }
    if ($t5xxl_device) {
      [void]$ext_args.Add("--t5xxl_device=$t5xxl_device")
    }
    if ($t5xxl_dtype) {
      [void]$ext_args.Add("--t5xxl_dtype=$t5xxl_dtype")
    }
    if ($text_encoder_batch_size) {
      [void]$ext_args.Add("--text_encoder_batch_size=$text_encoder_batch_size")
    }
  }
  if ($cache_text_encoder_outputs -ne 0) { 
    [void]$ext_args.Add("--cache_text_encoder_outputs")
    if ($cache_text_encoder_outputs_to_disk -ne 0) { 
      [void]$ext_args.Add("--cache_text_encoder_outputs_to_disk")
    }
    $shuffle_caption = 0
    $loraplus_text_encoder_lr_ratio = 0
    $caption_dropout_rate = 0
    $caption_tag_dropout_rate = 0
  }
  [void]$ext_args.Add("--sdpa")
}
elseif ($torch_compile) {
  [void]$ext_args.Add("--sdpa")
  [void]$ext_args.Add("--torch_compile")
  if ($dynamo_backend) {
    [void]$ext_args.Add("--dynamo_backend=$dynamo_backend")
  }
  if ($TORCHINDUCTOR_FX_GRAPH_CACHE -ne 0) {
    $Env:TORCHINDUCTOR_FX_GRAPH_CACHE = "1"
    if ($TORCHINDUCTOR_CACHE_DIR) {
      $Env:TORCHINDUCTOR_CACHE_DIR = $TORCHINDUCTOR_CACHE_DIR
    }
  }
}
else {
  [void]$ext_args.Add("--xformers")
}

if ($train_mode -ilike "stable_cascade*") {
  if ($effnet_checkpoint_path) {
    [void]$ext_args.Add("--effnet_checkpoint_path=$effnet_checkpoint_path")
  }
  if ($stage_c_checkpoint_path) {
    [void]$ext_args.Add("--stage_c_checkpoint_path=$stage_c_checkpoint_path")
  }
  if ($text_model_checkpoint_path) {
    [void]$ext_args.Add("--text_model_checkpoint_path=$text_model_checkpoint_path")
  }
  if ($save_text_model -ne 0) {
    [void]$ext_args.Add("--save_text_model")
  }
  if ($previewer_checkpoint_path) {
    [void]$ext_args.Add("--previewer_checkpoint_path=$previewer_checkpoint_path")
  }
  if ($adaptive_loss_weight -ne 0) {
    [void]$ext_args.Add("--adaptive_loss_weight")
  }
}

if ($dataset_class) { 
  [void]$ext_args.Add("--dataset_class=$dataset_class")
}
elseif ($dataset_config) {
  [void]$ext_args.Add("--dataset_config=$dataset_config")
}
else {
  [void]$ext_args.Add("--train_data_dir=$train_data_dir")
  if ($reg_data_dir) {
    [void]$ext_args.Add("--reg_data_dir=$reg_data_dir")
  }
  if ($batch_size) {
    [void]$ext_args.Add("--train_batch_size=$batch_size")
  }
  if ($resolution) {
    [void]$ext_args.Add("--resolution=$resolution")
  }
  if ($enable_bucket) {
    [void]$ext_args.Add("--enable_bucket")
    [void]$ext_args.Add("--min_bucket_reso=$min_bucket_reso")
    [void]$ext_args.Add("--max_bucket_reso=$max_bucket_reso")
    if ($bucket_no_upscale) {
      [void]$ext_args.Add("--bucket_no_upscale")
    }
  }
}

if ($caption_tag_dropout_rate) {
  [void]$ext_args.Add("--caption_tag_dropout_rate=$caption_tag_dropout_rate")
}

if ($pretrained_model) {
  [void]$ext_args.Add("--pretrained_model_name_or_path=$pretrained_model")
}

if ($vae) {
  [void]$ext_args.Add("--vae=$vae")
}

if ($disable_mmap_load_safetensors) {
  [void]$ext_args.Add("--disable_mmap_load_safetensors")
}

if ($save_model_as) {
  [void]$ext_args.Add("--save_model_as=$save_model_as")
}

if ($is_v2_model) {
  [void]$ext_args.Add("--v2")
  $min_snr_gamma = 0
  $debiased_estimation_loss = 0
  if ($v_parameterization) {
    [void]$ext_args.Add("--v_parameterization")
    [void]$ext_args.Add("--scale_v_pred_loss_like_noise_pred")
    [void]$ext_args.Add("--zero_terminal_snr")
  }
}
elseif ($train_mode -ilike "hunyuan*") {
  $laungh_script = "hunyuan_" + $laungh_script
  $min_snr_gamma = 0
  $debiased_estimation_loss = 0
  [void]$ext_args.Add("--v_parameterization")
  [void]$ext_args.Add("--scale_v_pred_loss_like_noise_pred")
  [void]$ext_args.Add("--zero_terminal_snr")
}
else {
  [void]$ext_args.Add("--clip_skip=$clip_skip")
}

if ($prior_loss_weight -and $prior_loss_weight -ne 1) {
  [void]$ext_args.Add("--prior_loss_weight=$prior_loss_weight")
}

if ($network_dim) {
  [void]$ext_args.Add("--network_dim=$network_dim")
}

if ($network_alpha) {
  [void]$ext_args.Add("--network_alpha=$network_alpha")
}

if ($training_comment) {
  [void]$ext_args.Add("--training_comment=$training_comment")
}

if ($persistent_workers) {
  [void]$ext_args.Add("--persistent_data_loader_workers")
}

if ($max_data_loader_n_workers) {
  [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}

if ($shuffle_caption) {
  [void]$ext_args.Add("--shuffle_caption")
}

if ($weighted_captions) {
  [void]$ext_args.Add("--weighted_captions")
}

if ($cache_latents) { 
  [void]$ext_args.Add("--cache_latents")
  if ($cache_latents_to_disk) {
    [void]$ext_args.Add("--cache_latents_to_disk")
  }
}

if ($output_config) {
  [void]$ext_args.Add("--output_config")
  [void]$ext_args.Add("--config_file=$config_file")
}

if ($gradient_checkpointing) {
  [void]$ext_args.Add("--gradient_checkpointing")
}

if ($save_state -eq 1) {
  [void]$ext_args.Add("--save_state")
  if ($save_state_on_train_end -eq 1) {
    [void]$ext_args.Add("--save_state_on_train_end")
  }
}

if ($resume) {
  [void]$ext_args.Add("--resume=$resume")
}

if ($noise_offset -ne 0) {
  [void]$ext_args.Add("--noise_offset=$noise_offset")
  if ($adaptive_noise_scale) {
    [void]$ext_args.Add("--adaptive_noise_scale=$adaptive_noise_scale")
  }
  if ($noise_offset_random_strength) {
    [void]$ext_args.Add("--noise_offset_random_strength")
  }
}
elseif ($multires_noise_iterations -ne 0) {
  [void]$ext_args.Add("--multires_noise_iterations=$multires_noise_iterations")
  [void]$ext_args.Add("--multires_noise_discount=$multires_noise_discount")
}

if ($network_dropout -ne 0) {
  $enable_lycoris = 0
  [void]$ext_args.Add("--network_dropout=$network_dropout")
  if ($scale_weight_norms -ne 0) { 
    [void]$ext_args.Add("--scale_weight_norms=$scale_weight_norms")
  }
  if ($enable_dylora -ne 0) {
    [void]$ext_args.Add("--network_args")
    if ($rank_dropout) {
      [void]$ext_args.Add("rank_dropout=$rank_dropout")
    }
    if ($module_dropout) {
      [void]$ext_args.Add("module_dropout=$module_dropout")
    }
  }
}

if ($enable_block_weights) {
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("down_lr_weight=$down_lr_weight")
  [void]$ext_args.Add("mid_lr_weight=$mid_lr_weight")
  [void]$ext_args.Add("up_lr_weight=$up_lr_weight")
  [void]$ext_args.Add("block_lr_zero_threshold=$block_lr_zero_threshold")
  if ($enable_block_dim) {
    [void]$ext_args.Add("block_dims=$block_dims")
    [void]$ext_args.Add("block_alphas=$block_alphas")
    if ($conv_block_dims) {
      [void]$ext_args.Add("conv_block_dims=$conv_block_dims")
      if ($conv_block_alphas) {
        [void]$ext_args.Add("conv_block_alphas=$conv_block_alphas")
      }
    }
    elseif ($conv_dim) {
      [void]$ext_args.Add("conv_dim=$conv_dim")
      if ($conv_alpha) {
        [void]$ext_args.Add("conv_alpha=$conv_alpha")
      }
    }
  }
}
elseif ($enable_lycoris) {
  $network_module = "lycoris.kohya"
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("algo=$algo")
  if ($algo -ine "ia3" -and $algo -ine "diag-oft") {
    if ($algo -ine "full") {
      if ($conv_dim) {
        [void]$ext_args.Add("conv_dim=$conv_dim")
        if ($conv_alpha) {
          [void]$ext_args.Add("conv_alpha=$conv_alpha")
        }
      }
      if ($use_tucker) {
        [void]$ext_args.Add("use_tucker=True")
      }
      if ($algo -ine "dylora") {
        if ($dora_wd) {
          [void]$ext_args.Add("dora_wd=True")
        }
        if ($bypass_mode) {
          [void]$ext_args.Add("bypass_mode=True")
        }
        if ($use_scalar) {
          [void]$ext_args.Add("use_scalar=True")
        }
      }
    }
    [void]$ext_args.Add("preset=$preset")
  }
  if ($dropout -and $algo -ieq "locon") {
    [void]$ext_args.Add("dropout=$dropout")
  }
  if ($train_norm -and $algo -ine "ia3") {
    [void]$ext_args.Add("train_norm=True")
  }
  if ($algo -ieq "lokr") {
    [void]$ext_args.Add("factor=$factor")
    if ($decompose_both) {
      [void]$ext_args.Add("decompose_both=True")
    }
    if ($full_matrix) {
      [void]$ext_args.Add("full_matrix=True")
    }
  }
  elseif ($algo -ieq "dylora") {
    [void]$ext_args.Add("block_size=$block_size")
  }
  elseif ($algo -ieq "diag-oft") {
    if ($rescaled) {
      [void]$ext_args.Add("rescaled=True")
    }
    if ($constrain) {
      [void]$ext_args.Add("constrain=$constrain")
    }
  }
}
elseif ($enable_dylora) {
  $network_module = "networks.dylora"
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("unit=$unit")
  if ($conv_dim) {
    [void]$ext_args.Add("conv_dim=$conv_dim")
    if ($conv_alpha) {
      [void]$ext_args.Add("conv_alpha=$conv_alpha")
    }
  }
  if ($module_dropout) {
    [void]$ext_args.Add("module_dropout=$module_dropout")
  }
}
elseif ($enable_lora_fa) {
  $network_module = "networks.lora_fa"
}
elseif ($enable_oft) {
  $network_module = "networks.oft"
}
else {
  if ($conv_dim) {
    [void]$ext_args.Add("--network_args")
    [void]$ext_args.Add("conv_dim=$conv_dim")
    if ($conv_alpha) {
      [void]$ext_args.Add("conv_alpha=$conv_alpha")
    }
  }
}

if ($optimizer_type -ieq "adafactor") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("scale_parameter=False")
  [void]$ext_args.Add("warmup_init=False")
  [void]$ext_args.Add("relative_step=False")
  if ($lr_scheduler -and $lr_scheduler -ine "constant") {
    $lr_warmup_steps = 100
  }
}

if ($optimizer_type -ilike "DAdapt*") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  if ($optimizer_type -ieq "DAdaptation" -or $optimizer_type -ilike "DAdaptAdam*") {
    [void]$ext_args.Add("decouple=True")
    if ($optimizer_type -ieq "DAdaptAdam") {
      [void]$ext_args.Add("use_bias_correction=True")
    }
  }
  $lr = "1"
  if ($unet_lr) {
    $unet_lr = $lr
  }
  if ($text_encoder_lr) {
    $text_encoder_lr = $lr
  }
}

if ($optimizer_type -ieq "Lion" -or $optimizer_type -ieq "Lion8bit" -or $optimizer_type -ieq "PagedLion8bit") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.95,.98")
}

if ($optimizer_type -ieq "AdamW8bit") {
  $optimizer_type = ""
  [void]$ext_args.Add("--use_8bit_adam")
}

if ($optimizer_type -ieq "PagedAdamW8bit" -or $optimizer_type -ieq "AdamW") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.9,.95")
}

if ($optimizer_type -ieq "Sophia") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SophiaH")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "Prodigy") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.9,.99")
  [void]$ext_args.Add("decouple=True")
  [void]$ext_args.Add("use_bias_correction=True")
  [void]$ext_args.Add("d_coef=$d_coef")
  if ($lr_warmup_steps) {
    [void]$ext_args.Add("safeguard_warmup=True")
  }
  if ($d0) {
    [void]$ext_args.Add("d0=$d0")
  }
  $lr = "1"
  if ($unet_lr) {
    $unet_lr = $lr
  }
  if ($text_encoder_lr) {
    $text_encoder_lr = $lr
  }
}

if ($optimizer_type -ieq "Ranger") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "Adan") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "StableAdamW") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("decouple_lr=True")
  }
}

if ($optimizer_type -ieq "Tiger") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Tiger")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ilike "*ScheduleFree") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.08")
  [void]$ext_args.Add("weight_lr_power=0")
}

if ($optimizer_type -ieq "adammini") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "adamg") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.1")
  [void]$ext_args.Add("weight_decouple=True")
}

if ($optimizer_type -ieq "came") {
  [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.CAME")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "sara") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("threshold=2e-3")
}

if ($gradfilter_ema_alpha -ne 0) {
  [void]$ext_args.Add("--gradfilter_ema_alpha=$gradfilter_ema_alpha")
  [void]$ext_args.Add("--gradfilter_ema_lamb=$gradfilter_ema_lamb")
}

if ($unet_lr) {
  if ($train_unet_only) {
    $train_text_encoder_only = 0
    $loraplus_text_encoder_lr_ratio = 0
    [void]$ext_args.Add("--network_train_unet_only")
  }
  [void]$ext_args.Add("--unet_lr=$unet_lr")
}

if ($text_encoder_lr) {
  if ($train_text_encoder_only) {
    $loraplus_unet_lr_ratio = 0
    [void]$ext_args.Add("--network_train_text_encoder_only")
  }
  [void]$ext_args.Add("--text_encoder_lr=$text_encoder_lr")
}

if ($enable_lora_plus) {
  [void]$ext_args.Add("--network_args")
  if ($loraplus_unet_lr_ratio) {
    [void]$ext_args.Add("loraplus_lr_ratio=$loraplus_unet_lr_ratio")
  }
  elseif ($loraplus_text_encoder_lr_ratio -eq 0) {
    [void]$ext_args.Add("loraplus_lr_ratio=$loraplus_lr_ratio")
  }
  if ($loraplus_text_encoder_lr_ratio) {
    [void]$ext_args.Add("loraplus_text_encoder_lr_ratio=$loraplus_text_encoder_lr_ratio")
  }
}

if ($network_weights) {
  [void]$ext_args.Add("--network_weights=$network_weights")
}

if ($network_multiplier -ne 1.0) {
  [void]$ext_args.Add("--network_multiplier=$network_multiplier")
}

if ($keep_tokens) {
  [void]$ext_args.Add("--keep_tokens=$keep_tokens")
}

if ($keep_tokens_separator) {
  [void]$ext_args.Add("--keep_tokens_separator=$keep_tokens_separator")
}

if ($secondary_separator) {
  [void]$ext_args.Add("--secondary_separator=$secondary_separator")
}

if ($enable_wildcard) {
  [void]$ext_args.Add("--enable_wildcard")
}

if ($caption_prefix) {
  [void]$ext_args.Add("--caption_prefix=$caption_prefix")
}

if ($caption_suffix) {
  [void]$ext_args.Add("--caption_suffix=$caption_suffix")
}

if ($alpha_mask) {
  [void]$ext_args.Add("--alpha_mask")
}

if ($min_snr_gamma -ne 0) {
  [void]$ext_args.Add("--min_snr_gamma=$min_snr_gamma")
}
elseif ($debiased_estimation_loss -ne 0) {
  [void]$ext_args.Add("--debiased_estimation_loss")
}

if ($loss_type -ne "l2") {
  [void]$ext_args.Add("--loss_type=$loss_type")
  if ($huber_schedule -ne "snr") {
    [void]$ext_args.Add("--huber_schedule=$huber_schedule")
  }
  if ($huber_c -ne 0.1) {
    [void]$ext_args.Add("--huber_c=$huber_c")
  }
}

if ($immiscible_noise) {
  [void]$ext_args.Add("--immiscible_noise")
}

if ($ip_noise_gamma -ne 0) {
  [void]$ext_args.Add("--ip_noise_gamma=$ip_noise_gamma")
  if ($ip_noise_gamma_random_strength) {
    [void]$ext_args.Add("--ip_noise_gamma_random_strength")
  }
}

if ($wandb_api_key) {
  [void]$ext_args.Add("--wandb_api_key=$wandb_api_key")
  [void]$ext_args.Add("--log_with=wandb")
  [void]$ext_args.Add("--log_tracker_name=" + $output_name)
}

if ($enable_sample) {
  if ($sample_at_first) {
    [void]$ext_args.Add("--sample_at_first")
  }
  [void]$ext_args.Add("--sample_every_n_epochs=$sample_every_n_epochs")
  [void]$ext_args.Add("--sample_prompts=$sample_prompts")
  [void]$ext_args.Add("--sample_sampler=$sample_sampler")
}

if ($base_weights) {
  [void]$ext_args.Add("--base_weights")
  foreach ($base_weight in $base_weights.Split(" ")) {
    [void]$ext_args.Add($base_weight)
  }
  [void]$ext_args.Add("--base_weights_multiplier")
  foreach ($ratio in $base_weights_multiplier.Split(" ")) {
    [void]$ext_args.Add([float]$ratio)
  }
}

if ($fused_backward_pass -ne 0) {
  [void]$ext_args.Add("--fused_backward_pass")
  $gradient_accumulation_steps = 0
  $full_fp16 = 0
  $full_bf16 = 0
  $fp8_base = 0
  $fp8_base_unet = 0
  $mixed_precision = ""
  $save_precision = "fp16"
}
elseif ($fused_optimizer_groups) {
  [void]$ext_args.Add("--fused_optimizer_groups")
}

if ($fp8_base -ne 0) {
  [void]$ext_args.Add("--fp8_base")
}
if ($fp8_base_unet -ne 0) {
  [void]$ext_args.Add("--fp8_base_unet")
}
if ($full_fp16 -ne 0) {
  [void]$ext_args.Add("--full_fp16")
  $mixed_precision = "fp16"
  $save_precision = "fp16"
}
elseif ($full_bf16 -ne 0) {
  [void]$ext_args.Add("--full_bf16")
  $mixed_precision = "bf16"
  $save_precision = "bf16"
}

if ($mixed_precision) {
  [void]$ext_args.Add("--mixed_precision=$mixed_precision")
}

if ($network_module) {
  [void]$ext_args.Add("--network_module=$network_module")
}

if ($gradient_accumulation_steps) {
  [void]$ext_args.Add("--gradient_accumulation_steps=$gradient_accumulation_steps")
}

if ($optimizer_accumulation_steps) {
  [void]$ext_args.Add("--optimizer_accumulation_steps=$optimizer_accumulation_steps")
}

if ($lr_scheduler) {
  [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}

if ($lr_scheduler_num_cycles) {
  [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}

if ($lr_warmup_steps) {
  if ($gradient_accumulation_steps) {
    $lr_warmup_steps = $lr_warmup_steps * $gradient_accumulation_steps
  }
  [void]$ext_args.Add("--lr_warmup_steps=$lr_warmup_steps")
}

if ($lr_decay_steps) {
  if ($gradient_accumulation_steps) {
    $lr_decay_steps = $lr_decay_steps * $gradient_accumulation_steps
  }
  [void]$ext_args.Add("--lr_decay_steps=$lr_decay_steps")
}

if ($lr_scheduler_timescale) {
  [void]$ext_args.Add("--lr_scheduler_timescale=$lr_scheduler_timescale")
}

if ($lr_scheduler_min_lr_ratio) {
  [void]$ext_args.Add("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio")
}

if ($caption_dropout_every_n_epochs) {
  [void]$ext_args.Add("--caption_dropout_every_n_epochs=$caption_dropout_every_n_epochs")
}
if ($caption_dropout_rate) {
  [void]$ext_args.Add("--caption_dropout_rate=$caption_dropout_rate")
}
if ($caption_tag_dropout_rate) {
  [void]$ext_args.Add("--caption_tag_dropout_rate=$caption_tag_dropout_rate")
}

# run train
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 $launch_args "./sd-scripts/$laungh_script.py" `
  --output_dir="./output" `
  --logging_dir="./logs" `
  --max_train_epochs=$max_train_epoches `
  --learning_rate=$lr `
  --output_name=$output_name `
  --save_every_n_epochs=$save_every_n_epochs `
  --save_precision=$save_precision `
  --seed=$seed  `
  --max_token_length=225 `
  --caption_extension=".txt" `
  --vae_batch_size=$vae_batch_size `
  $ext_args

Write-Output "Train finished"

Read-Host | Out-Null ;