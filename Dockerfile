# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:13:26 2026

@author: User
"""

FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
EXPOSE 8000
EXPOSE 8501
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]