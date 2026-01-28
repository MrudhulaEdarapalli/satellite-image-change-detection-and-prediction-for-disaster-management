import os, base64, re

app_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'app.py')
bg_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'bg.png')

if os.path.exists(bg_path):
    with open(bg_path, 'rb') as f:
        bg_base64 = base64.b64encode(f.read()).decode()
else:
    bg_base64 = ""

with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update background in stAppViewContainer
new_bg_css = f'''
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("data:image/png;base64,{bg_base64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }}
'''

# Use broader regex to match both single and double quotes for data-testid
content = re.sub(r'\[data-testid=[\'\"]stAppViewContainer[\'\"]\].*?\{.*?\}', new_bg_css, content, flags=re.DOTALL)

# 2. Reduce title size
content = content.replace('font-size: 4rem;', 'font-size: 2.5rem;')

# 3. Add CSS to hide toolbar, header, footer
hide_css = '''
    /* Hide Streamlit UI elements */
    header {visibility: hidden !important; height: 0 !important;}
    footer {visibility: hidden !important;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
'''
if '</style>' in content:
    content = content.replace('</style>', hide_css + '\n    </style>')

# 4. Remove manual footer section
content = re.sub(r'# --- Footer Section ---.*?unsafe_allow_html=True\)', '', content, flags=re.DOTALL)

# 5. Clean up redundant .main background
# This regex targets the long base64 one
content = re.sub(r'\.main\s*\{\s*background: url\(data:image/png;base64,.*?\).*?\}', '.main { background: transparent !important; }', content, flags=re.DOTALL)

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(content)
