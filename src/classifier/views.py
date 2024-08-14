from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import APIUsage
from datetime import timedelta
from django.utils import timezone
import torch
from transformers import BertTokenizer
from .model import TrademarkClassifier, predict_class

class TrademarkClassPredictionView(APIView):
    def post(self, request, *args, **kwargs):
        user_id = request.data.get('user_id')
        description = request.data.get('description')

        if not user_id or not description:
            return Response({"error": "user_id and description are required."}, status=status.HTTP_400_BAD_REQUEST)

        usage_record, created = APIUsage.objects.get_or_create(user_id=user_id)

        if usage_record.request_count >= 5:
            return Response({"error": "Rate limit exceeded."}, status=status.HTTP_429_TOO_MANY_REQUESTS)

        usage_record.request_count += 1
        usage_record.save()

        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TrademarkClassifier(n_classes=45)
        model.load_state_dict(torch.load('../model/trademark_model.pt'))
        model = model.to(device)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        class_id = predict_class(description, model, tokenizer, device)

        return Response({"class_id": class_id}, status=status.HTTP_200_OK)
