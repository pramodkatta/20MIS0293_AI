from django.db import models

class APIUsage(models.Model):
    user_id = models.CharField(max_length=100)
    request_count = models.IntegerField(default=1)
    last_request_time = models.DateTimeField(auto_now=True)
