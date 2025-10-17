@echo off
echo Installing libraries to virtual environment...
echo.

".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Checking installed libraries...
echo.

echo Checking numpy...
".venv\Scripts\python.exe" -c "import numpy; print('numpy:', numpy.__version__)" 2>nul || echo numpy: NOT INSTALLED

echo Checking scikit-learn...
".venv\Scripts\python.exe" -c "import sklearn; print('scikit-learn:', sklearn.__version__)" 2>nul || echo scikit-learn: NOT INSTALLED

echo Checking matplotlib...
".venv\Scripts\python.exe" -c "import matplotlib; print('matplotlib:', matplotlib.__version__)" 2>nul || echo matplotlib: NOT INSTALLED

echo Checking opencv-python...
".venv\Scripts\python.exe" -c "import cv2; print('opencv-python:', cv2.__version__)" 2>nul || echo opencv-python: NOT INSTALLED

echo Checking Pillow...
".venv\Scripts\python.exe" -c "import PIL; print('Pillow:', PIL.__version__)" 2>nul || echo Pillow: NOT INSTALLED

echo Checking joblib...
".venv\Scripts\python.exe" -c "import joblib; print('joblib:', joblib.__version__)" 2>nul || echo joblib: NOT INSTALLED

echo.
echo Done!
pause