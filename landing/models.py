from django.db import models
import json

class Prediction(models.Model):
    input_data = models.TextField()  # Store as JSON string
    result = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)

    def input_dict(self):
        return json.loads(self.input_data) 