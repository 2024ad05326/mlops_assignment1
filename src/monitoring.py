from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

PREDICTION_COUNTER = Counter(
    'heart_disease_predictions_total',
    'Total number of predictions made',
    ['status', 'prediction']
)
PREDICTION_LATENCY = Histogram(
    'heart_disease_prediction_latency_seconds',
    'Latency of prediction requests in seconds'
)
CONFIDENCE_GAUGE = Gauge(
    'heart_disease_prediction_confidence',
    'Confidence score of the latest prediction'
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def start_metrics_server(port: int = 8001):
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")
