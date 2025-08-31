# tagger script by @bdsqlsz ���ϰ�
# �������ڱ�ǩ�ļ���ͷ���ָ���ַ����Ĺ���

# Train data path
$chufaci = "houtu," # ���޸�Ϊʵ����Ҫ��ӵ� ������+Ӣ�Ķ���
$train_data_dir = "./train/houtu/20_houtu" # input images path | ͼƬ����·��
$repo_id = "SmilingWolf/wd-eva02-large-tagger-v3" # model repo id from huggingface |huggingfaceģ��repoID
$model_dir = "wd14_tagger_model" # model dir path | ����ģ���ļ���·��
$batch_size = 12 # batch size in inference �������С��Խ��Խ��
$max_data_loader_n_workers = 0 # enable image reading by DataLoader with this number of workers (faster) | 0���
$thresh = 0.27 # concept thresh | ��Сʶ����ֵ
$general_threshold = 0.27 # general threshold | ����ʶ����ֵ 
$character_threshold = 0.3 # character threshold | ��������ʶ����ֵ
$recursive = 1 # search for images in subfolders recursively | �ݹ������²��ļ��У�1Ϊ����0Ϊ��
$frequency_tags = 0 # order by frequency tags | �Ӵ�С��ʶ���������ǩ��1Ϊ����0Ϊ��
$onnx = 1 #ʹ��ONNXģ��


#Tag Edit | ��ǩ�༭
$remove_underscore = 1 # remove_underscore | �»���ת�ո�1Ϊ����0Ϊ�� 
$undesired_tags = "simple background" # no need tags | �ų���ǩ
$use_rating_tags = 0 #ʹ�����ֱ�ǩ
$use_rating_tags_as_last_tag= 0 #�����ǩ�����
$character_tags_first = 1 #��ɫ��ǩ����ǰ��
$character_tag_expand = 1 #���� ϵ�в�֣�chara_name_(series) ��Ϊ chara_name, series.
$always_first_tags = "1girl,1boy,2girls,2boys,3girls,3boys" #ָ����ǩ����ǰ����ͼ���г���ĳ����ǩʱ������������ñ�ǩ������ָ�������ǩ���Զ��ŷָ�
$tag_replacement = "" #ִ�б���滻��ָ����ʽΪ tag1,tag2;tag3,tag4�����ʹ�� , �� ;������\ת�塣���磬ָ�� aira tsubase,aira tsubase��uniform��������Ҫѵ���ض���װʱ����aira tsubase,aira tsubase\, heir of shadows������ǩ�в�����ϵ������ʱ����
$remove_parents_tag = 0

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
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

if ($repo_id) {
  [void]$ext_args.Add("--repo_id=$repo_id")
}

if ($model_dir) {
  [void]$ext_args.Add("--model_dir=$model_dir")
}

if ($batch_size) {
  [void]$ext_args.Add("--batch_size=$batch_size")
}

if ($max_data_loader_n_workers) {
  [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}

if ($general_threshold) {
  [void]$ext_args.Add("--general_threshold=$general_threshold")
}

if ($character_threshold) {
  [void]$ext_args.Add("--character_threshold=$character_threshold")
}

if ($remove_underscore) {
  [void]$ext_args.Add("--remove_underscore")
}

if ($undesired_tags) {
  [void]$ext_args.Add("--undesired_tags=$undesired_tags")
}

if ($recursive) {
  [void]$ext_args.Add("--recursive")
}

if ($frequency_tags) {
  [void]$ext_args.Add("--frequency_tags")
}

if ($onnx) {
  [void]$ext_args.Add("--onnx")
}

if ($character_tags_first) {
  [void]$ext_args.Add("--character_tags_first")
}

if ($character_tag_expand) {
  [void]$ext_args.Add("--character_tag_expand")
}

if ($use_rating_tags) {
  [void]$ext_args.Add("--use_rating_tags")
  if ($use_rating_tags_as_last_tag) {
    [void]$ext_args.Add("--use_rating_tags_as_last_tag")
  }
}

if ($always_first_tags) {
  [void]$ext_args.Add("--always_first_tags=$always_first_tags")
}

if ($tag_replacement) {
  [void]$ext_args.Add("--tag_replacement=$tag_replacement")
}

if ($remove_parents_tag) {
  [void]$ext_args.Add("--remove_parents_tag")
}

# run tagger
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/finetune/tag_images_by_wd14_tagger.py" `
  $train_data_dir `
  --thresh=$thresh `
  --caption_extension .txt `
  $ext_args

Write-Output "Tagger finished"

# �����������б�ǩ�ļ���ͷ���ָ���ַ���
Write-Output "��ʼ�ڱ�ǩ�ļ���ͷ���ָ������..."

# ���Ŀ¼�Ƿ����
if (-not (Test-Path -Path $train_data_dir -PathType Container)) {
    Write-Error "����: Ŀ¼ '$train_data_dir' ������"
    Read-Host | Out-Null
    exit 1
}

# ����ʼ�ַ����Ƿ�������
if ([string]::IsNullOrEmpty($chufaci)) {
    Write-Warning "����: ���� `$chufaci δ�����Ϊ�գ�������Ӳ���"
    Read-Host | Out-Null
    exit 0
}

# ��ȡ����TXT�ļ�������recursive���������Ƿ������Ŀ¼��
if ($recursive) {
    $txtFiles = Get-ChildItem -Path $train_data_dir -Filter *.txt -Recurse -File
} else {
    $txtFiles = Get-ChildItem -Path $train_data_dir -Filter *.txt -File
}

if ($txtFiles.Count -eq 0) {
    Write-Warning "��Ŀ¼ '$train_data_dir' ��δ�ҵ��κ�TXT�ļ�"
    Read-Host | Out-Null
    exit 0
}

# ����ÿ��TXT�ļ����ڿ�ͷ�������
foreach ($file in $txtFiles) {
    try {
        # ��ȡ�ļ�����
        $content = Get-Content -Path $file.FullName -Raw
        
        # ����Ƿ��Ѿ���ӹ��������ظ���ӣ�
        if (-not $content.StartsWith($chufaci)) {
            # �����ݿ�ͷ����ַ���
            $newContent = $chufaci + $content
            
            # д���ļ�
            Set-Content -Path $file.FullName -Value $newContent -Force
            
            Write-Host "�Ѵ����ļ�: $($file.FullName)"
        } else {
            Write-Host "�ļ��Ѱ�����ͷ���ݣ�����: $($file.FullName)"
        }
    }
    catch {
        Write-Error "�����ļ� '$($file.FullName)' ʱ����: $_"
    }
}

Write-Host "`n������ɣ��������� $($txtFiles.Count) ���ļ�"
Read-Host | Out-Null
    