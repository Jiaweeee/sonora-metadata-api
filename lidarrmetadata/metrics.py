from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from quart import request

class PrometheusMetrics:
    def __init__(self, app=None):
        # 请求计数器，按路径和状态码区分
        self.request_counter = Counter(
            'http_requests_total', 
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        # 请求耗时直方图
        self.request_latency = Histogram(
            'http_request_duration_seconds', 
            'HTTP request latency in seconds',
            ['method', 'endpoint'],
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
        )
        
        # 当前运行中的请求数量
        self.in_progress = Gauge(
            'http_requests_in_progress',
            'Number of HTTP requests in progress',
            ['method', 'endpoint']
        )
        
        # 请求大小
        self.request_size = Summary(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint']
        )
        
        # 响应大小
        self.response_size = Summary(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint', 'status']
        )
        
        # 错误计数器
        self.error_counter = Counter(
            'http_request_errors_total',
            'Total number of HTTP request errors',
            ['method', 'endpoint', 'exception']
        )
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """初始化应用，添加请求中间件"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
    async def before_request(self):
        """请求前的处理"""
        request.start_time = time.time()
        endpoint = request.endpoint or 'unknown'
        method = request.method
        
        # 增加运行中的请求计数
        self.in_progress.labels(method=method, endpoint=endpoint).inc()
        
        # 记录请求大小
        content_length = request.content_length or 0
        self.request_size.labels(method=method, endpoint=endpoint).observe(content_length)
    
    async def after_request(self, response):
        """请求后的处理"""
        endpoint = request.endpoint or 'unknown'
        method = request.method
        status = response.status_code
        
        # 减少运行中的请求计数
        self.in_progress.labels(method=method, endpoint=endpoint).dec()
        
        # 统计请求耗时
        latency = time.time() - request.start_time
        self.request_latency.labels(method=method, endpoint=endpoint).observe(latency)
        
        # 统计请求数
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        
        # 记录响应大小
        content_length = int(response.headers.get('Content-Length', 0))
        self.response_size.labels(method=method, endpoint=endpoint, status=status).observe(content_length)
        
        return response
    
    def record_exception(self, exception):
        """记录异常"""
        try:
            endpoint = request.endpoint or 'unknown'
            method = request.method
            
            self.error_counter.labels(
                method=method, 
                endpoint=endpoint, 
                exception=type(exception).__name__
            ).inc()
        except Exception:
            # 防止在没有请求上下文时出错
            pass

# 创建单例实例
metrics = PrometheusMetrics() 