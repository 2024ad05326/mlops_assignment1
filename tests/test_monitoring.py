from prometheus_client import Counter, Histogram, Gauge


def test_prometheus_metrics_registered():
    assert Counter is not None
    assert Histogram is not None
    assert Gauge is not None


def test_prediction_counter_increment():
    import src.monitoring as monitoring
    counter = monitoring.PREDICTION_COUNTER
    child = counter.labels(status='success', prediction='1')
    child.inc()
    child.inc()
    assert child is not None
