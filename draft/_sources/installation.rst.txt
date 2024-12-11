============
Installation
============

Python dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Modflow-obs requires `numpy`_, `pandas`_ and the `affine`_ library. Additionally, running the demonstration example requires Jupyter Notebooks. The easiest way to install these is using conda.

Instructions for installing `Miniforge <https://github.com/conda-forge/miniforge>`_, a minimal conda-based python distribution, can be found `here <https://github.com/DOI-USGS/python-for-hydrology/blob/main/installation/README.md#python-installation-instructions>`_.

The `requirements.yml` file included at the top level of this repository includes the necessary packages.

.. code-block:: bash

    conda env create -f requirements.yml

.. _affine: https://github.com/rasterio/affine
.. _numpy: https://numpy.org/
.. _pandas: http://pandas.pydata.org

Installing Modflow-obs using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pip can be used to fetch Modflow-obs directly from GitHub:

.. code-block:: bash

    pip install git+git://github.com/aleaf/modflow-obs@master

Subsequent updates can then be made with

.. code-block:: bash

    pip install --upgrade git+git://github.com/aleaf/modflow-obs@master

or by uninstalling and reinstalling. To uninstall:

.. code-block:: bash

    pip uninstall modflow-obs

Installing the Modflow-obs source code in-place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Alternatively, if you intend to contribute to Modflow-obs (please do!) or update your install frequently, the best route is probably to clone the source code from git and install it in place.

.. code-block:: bash

    git clone https://github.com/aleaf/modflow-obs.git
    cd modflow-obs
    pip install -e .

.. note::
    Don't forget the ``.`` after ``pip install -e``!

Your local copy of the Modflow-obs repository can then be subsequently updated with

.. code-block:: bash

    git pull origin master

.. note::
    If you are making local changes to Modflow-obs that you want to contribute, the workflow is slightly different. See the :ref:`Contributing to Modflow-obs` page for more details.
