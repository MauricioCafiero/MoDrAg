import torch
import os, re
import gradio as gr
import numpy as np
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from intake_function import intake, second_intake, start_models, full_tool_descriptions
from input_parsing import define_tool_hash


device = "cuda" if torch.cuda.is_available() else "cpu"

