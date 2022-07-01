.. _issues:

==========================
Known Issues & Limitations
==========================
We are aware that Conmo is still in a very early stage of development, so it is likely that as its use increases, various bugs will appear. 
Bugs that are detected will be published on this page in order to make it easier for users to prevent them. However, the Conmo development team is actively looking for and fixing any detected bugs.
Please, if you find a bug/issue that does not appear on this list, we would be grateful if you could email us at mym.inv.uniovi@gmail.com or post an issue on our Github.
Thanks in advance.

.. csv-table::
   :header: "Issue ID", "Severity", "Description"

   "001_split", "Low", "There are some problems with the use of Scikit-Learn's Time Series Splitter in the experiments. We are working on resolving them."
   "002_rul", "Medium", "rul_rve.py example seems to be failing during the metric calculation step."
   "003_tf", "Medium", "If your computer has one of the new Apple processors (M1 or M2) with ARM-based architecture, it is likely that when you try to use Conmo, the Tensorflow dependency will fail. To fix this temporarily you can install Conmo without dependencies: 'pip install --no-deps conmo' and then manually install the branch provided by Google for ARM architectures tensorflow-macos."