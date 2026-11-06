"""Local CLEAR-generated image/prompt pairs and native FLUX.2 caches."""
import hashlib
import json
import os
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image,ImageOps
from .checkpoint import atomic_json

BUCKETS_2K=((2048,2048),(2304,1728),(1728,2304),(2688,1536),(1536,2688),(2688,1792),(1792,2688))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def file_identity(path):
    p=Path(path).resolve();s=p.stat()
    return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)


def image_ids(h,w):
    return torch.cartesian_prod(torch.arange(1),torch.arange(h),torch.arange(w),torch.arange(1)).float()


def prompt_ids(length):
    ids=torch.zeros(length,4);ids[:,-1]=torch.arange(length)
    return ids


def prepare_image(path,size):
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        w,h=size;scale=max(w/image.width,h/image.height)
        image=image.resize((max(w,round(image.width*scale)),max(h,round(image.height*scale))),Image.Resampling.LANCZOS)
        x,y=(image.width-w)//2,(image.height-h)//2
        return image.crop((x,y,x+w,y+h))


def initialize_index(dataset_root,cache_root,weights_root):
    dataset_root=Path(dataset_root).resolve();root=Path(cache_root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    target=root/'manifest.json'
    if target.exists():
        manifest=json.loads(target.read_text())
        if manifest['model']!=model_identity(weights_root):raise ValueError('FLUX2 model changed; use a new cache root')
        if manifest['dataset_root']!=str(dataset_root):raise ValueError('Dataset root changed')
        return manifest
    rows=[];skipped=[]
    for record in json.loads((dataset_root/'prompts.json').read_text()):
        key=str(record['id']);image=dataset_root/'images'/f'{key}.png'
        if not image.is_file():continue # interrupted generation is not a training sample
        try:
            with Image.open(image) as im:w,h=im.size
            size=min(BUCKETS_2K,key=lambda s:abs(__import__('math').log((w/h)/(s[0]/s[1]))))
            if not record['prompt'].strip():raise ValueError('empty caption')
            rows.append(dict(id=key,caption=record['prompt'],image_path=str(image),size=list(size),
                             bucket=f'{size[0]}x{size[1]}',source=file_identity(image)))
        except Exception as exc:skipped.append(dict(id=key,error=str(exc)))
    if not rows:raise ValueError('No completed images in the requested dataset')
    manifest=dict(schema=1,dataset_root=str(dataset_root),model=model_identity(weights_root),
                  rows=rows,skipped=skipped,text=dict(native='Flux2KleinPipeline',layers=[9,18,27],length=512),
                  image=dict(buckets=BUCKETS_2K,vae='native mode, 2x2 packing, BN normalization',dtype='bfloat16'))
    manifest['fingerprint']=fingerprint(manifest)
    atomic_json(target,manifest)
    return manifest


def model_identity(root):
    root=Path(root)
    files=[root/'transformer/diffusion_pytorch_model.safetensors',root/'vae/diffusion_pytorch_model.safetensors',
           *sorted((root/'text_encoder').glob('*.safetensors')),root/'tokenizer/tokenizer_config.json']
    return [file_identity(p) for p in files]


def cache_path(root,kind,key):
    hashed=fingerprint(key)
    return Path(root)/kind/hashed[:2]/f'{hashed}.pt'


def valid(path,identity):
    try:
        meta=json.loads(path.with_suffix('.json').read_text());s=path.stat()
        return meta==dict(identity=identity,size=s.st_size,mtime_ns=s.st_mtime_ns)
    except (OSError,ValueError):return False


def atomic_tensor(path,value,identity):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.partial')
    torch.save({k:v.detach().cpu().contiguous() if torch.is_tensor(v) else v for k,v in value.items()},tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path);s=path.stat()
    atomic_json(path.with_suffix('.json'),dict(identity=identity,size=s.st_size,mtime_ns=s.st_mtime_ns))


class CachedDataset:
    def __init__(self,subset_root,cache_root,subset_size=0,limit=0):
        self.root=Path(cache_root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.fingerprint=self.manifest['fingerprint'];self.rows=self.manifest['rows'][:limit or None]
        completion=self.root/'complete.json'
        if not limit:
            if not completion.exists():raise FileNotFoundError('Native FLUX.2 preparation is incomplete')
            done=json.loads(completion.read_text())
            if done.get('fingerprint')!=self.fingerprint or done.get('samples')!=len(self.rows):
                raise ValueError('Cache completion identity mismatch')
        null=cache_path(self.root,'text','')
        if not valid(null,self.fingerprint):raise FileNotFoundError('Missing empty prompt cache')
        self.empty_prompt=torch.load(null,weights_only=True)
        self.bucket_indices=defaultdict(list)
        for i,row in enumerate(self.rows):self.bucket_indices[row['bucket']].append(i)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        if file_identity(row['image_path'])!=row['source']:raise ValueError(f'Source image changed: {row["id"]}')
        for kind,key in [('image',row['id']),('text',row['caption'])]:
            if not valid(cache_path(self.root,kind,key),self.fingerprint):raise RuntimeError(f'Missing/stale native FLUX.2 {kind} cache for {row["id"]}')
        image=torch.load(cache_path(self.root,'image',row['id']),weights_only=True)
        text=torch.load(cache_path(self.root,'text',row['caption']),weights_only=True)
        return dict(id=row['id'],**image,**text)
