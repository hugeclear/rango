# debug_dependencies.py
def test_dependency_check():
    """run_evaluation.pyと同じ方法でチェック"""
    
    # 方法1: importでチェック
    try:
        import sklearn
        print(f"✅ sklearn import: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ sklearn import failed: {e}")
    
    # 方法2: scikit-learnの名前でチェック
    try:
        import importlib
        sklearn_pkg = importlib.import_module('sklearn')
        print(f"✅ sklearn (importlib): {sklearn_pkg.__version__}")
    except ImportError as e:
        print(f"❌ sklearn (importlib) failed: {e}")
    
    # 方法3: pkgutilでチェック（一部のスクリプトが使用）
    import pkgutil
    packages = ['sklearn', 'scikit-learn', 'numpy', 'pandas', 'torch']
    
    for pkg in packages:
        if pkgutil.find_loader(pkg):
            print(f"✅ {pkg}: Found by pkgutil")
        else:
            print(f"❌ {pkg}: Not found by pkgutil")

if __name__ == "__main__":
    test_dependency_check()
