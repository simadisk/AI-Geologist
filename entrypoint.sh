# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:12:32 2026

@author: User
"""

#!/bin/bash
uvicorn api:app --host 0.0.0.0 --port 8000 &
streamlit run app.py --server.port 8501 --server.address 0.0.0.0