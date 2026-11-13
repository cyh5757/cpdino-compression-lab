"""Independent instance metrics and BF16 full-image/tile timing for CPDINO."""
import hashlib
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from cellpose import io, metrics, models

from .models import build_model
from .training import write_json
from .weights import load_weights


EVALUATION_SETTINGS = {
    'channel_axis':-1, 'normalize':True, 'diameter':None, 'batch_size':8,
    'bsize':384, 'tile_overlap':.1, 'cellprob_threshold':-1., 'flow_threshold':.4,
    'min_size':15, 'resample':True, 'niter':None, 'augment':False,
}


def _dense_labels(mask):
    mask=np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0 or not np.isfinite(mask).all() or (mask < 0).any():
        raise ValueError('Masks must be finite, nonnegative, nonempty 2D label arrays')
    if not np.equal(mask,np.floor(mask)).all():
        raise ValueError('Mask labels must be integers')
    ids=np.unique(mask)
    ids=ids[ids>0]
    dense=np.where(mask==0,0,np.searchsorted(ids,mask)+1).astype(np.int32)
    return dense,len(ids)


def instance_metrics(truth,prediction):
    if np.shape(truth)!=np.shape(prediction):
        raise ValueError('Ground-truth and prediction shapes differ')
    truth,ntrue=_dense_labels(truth)
    prediction,npred=_dense_labels(prediction)
    if ntrue and npred:
        ap,_,_,_=metrics.average_precision(truth,prediction,threshold=[.5,.75])
    else:
        ap=[float(ntrue==npred)]*2
    return {'ap50':float(ap[0]),'ap75':float(ap[1]),'gt_count':ntrue,
            'predicted_count':npred,'count_error':npred-ntrue}


def summarize_images(rows):
    if not rows:
        raise ValueError('Evaluation contains no images')
    return {'test_images':len(rows),'ap50':statistics.mean(row['ap50'] for row in rows),
            'ap75':statistics.mean(row['ap75'] for row in rows),
            'count_mae':statistics.mean(abs(row['count_error']) for row in rows),
            'mean_image_ms':statistics.mean(statistics.median(row['latency_ms']) for row in rows)}


def evaluate_checkpoint(checkpoint, samples, wrapper_checkpoint, output_dir, *,
                        backbone='vits', model_id='distilled_vits', device='cuda'):
    """Evaluate a selected checkpoint using the historical 64-image protocol.

    The wrapper checkpoint initializes Cellpose's public inference interface;
    its network is then replaced completely by the selected CPDINO network.
    Results with image identifiers remain in the caller's private run directory.
    """
    if torch.device(device).type != 'cuda':
        raise ValueError('The historical BF16 timing protocol requires CUDA')
    output_dir=Path(output_dir)
    if (output_dir/'result.json').exists():
        raise ValueError('An evaluation result already exists in this directory')
    if not samples or any(not sample.mask.is_file() for sample in samples):
        raise ValueError('Every test image requires an independent instance mask')
    checkpoint=Path(checkpoint)
    with checkpoint.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    net=build_model(backbone)
    load_weights(net,checkpoint,digest,scope='model')
    net.dtype=torch.bfloat16
    model=models.CellposeModel(device=torch.device(device),pretrained_model=str(wrapper_checkpoint),use_bfloat16=True)
    model.net=net.to(device).eval()
    model.backbone='dino_'+backbone
    output_dir.mkdir(parents=True,exist_ok=True)
    samples=list(samples)
    rows=[]
    with torch.inference_mode():
        tiles=torch.rand(8,3,384,384,device=device,dtype=torch.bfloat16)
        for _ in range(10):
            model.net(tiles)
        torch.cuda.synchronize(device)
        timings=[]
        for _ in range(30):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            model.net(tiles)
            end.record()
            timings.append((start,end))
        torch.cuda.synchronize(device)
        forward_ms=[start.elapsed_time(end) for start,end in timings]
        del tiles
        first=io.imread(str(samples[0].image))
        for _ in range(2):
            model.eval(first,**EVALUATION_SETTINGS)
        for sample in samples:
            image=io.imread(str(sample.image))
            truth=io.imread(str(sample.mask))
            latency=[]
            for _ in range(3):
                torch.cuda.synchronize(device)
                started=time.perf_counter()
                mask,_,_=model.eval(image,**EVALUATION_SETTINGS)
                torch.cuda.synchronize(device)
                latency.append((time.perf_counter()-started)*1000)
            rows.append({'image_id':sample.image.stem,'latency_ms':latency,**instance_metrics(truth,mask)})
            write_json(output_dir/'progress.json',{'completed':len(rows),'total':len(samples)})
    result={'model':model_id,'checkpoint_sha256':digest,'precision':'PyTorch BF16',
            'reported_device':torch.cuda.get_device_name(device),'settings':EVALUATION_SETTINGS,
            'tiles_per_second':8000/statistics.mean(forward_ms),'images':rows,**summarize_images(rows)}
    write_json(output_dir/'result.json',result)
    return result


def public_summary(result):
    """Export aggregate values without image identifiers or private paths."""
    keys=('model','checkpoint_sha256','precision','reported_device','settings',
          'tiles_per_second','test_images','ap50','ap75','count_mae','mean_image_ms')
    return {key:result[key] for key in keys}
