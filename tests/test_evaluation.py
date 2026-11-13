import numpy as np
import pytest

from cpdino_lab.evaluation import instance_metrics, summarize_images


def test_instance_metrics_accept_sparse_ids_and_match_geometry():
    truth=np.array([[0,4,4],[0,0,0],[9,9,0]])
    pred=np.array([[0,80,80],[0,0,0],[30,30,0]])
    row=instance_metrics(truth,pred)
    assert row['ap50']==1
    assert row['ap75']==1
    assert row['count_error']==0


def test_empty_masks_follow_original_evaluation_convention():
    empty=np.zeros((3,3),dtype=np.int32)
    assert instance_metrics(empty,empty)['ap50']==1
    assert instance_metrics(empty,np.ones_like(empty))['ap50']==0
    with pytest.raises(ValueError,match='shape'):
        instance_metrics(empty,empty[:2])


def test_summary_averages_per_image_medians_and_absolute_count_errors():
    rows=[{'ap50':.8,'ap75':.6,'count_error':-2,'latency_ms':[1,3,100]},
          {'ap50':.6,'ap75':.4,'count_error':4,'latency_ms':[10,20,30]}]
    result=summarize_images(rows)
    assert result['ap50']==pytest.approx(.7)
    assert result['ap75']==pytest.approx(.5)
    assert result['count_mae']==3
    assert result['mean_image_ms']==11.5
    assert result['test_images']==2
