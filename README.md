To run the script, please install specified libraries from requirements.txt 
and then simply run ``python main.py``

** IMPORTANT NOTE: The script may cause problems when running on Windows, 
and that is because of the pandarallel, if you have to run on windows, replace all calls for 
``df.parallel_apply`` to a simple ``df.apply`` in the script.   