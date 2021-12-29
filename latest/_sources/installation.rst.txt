============
Installation
============

Python dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Modflow-obs requires `numpy`_, `pandas`_ and the `affine`_ library. The easiest way to
install these is using conda and pip:

.. code-block:: bash

    conda install -c conda-forge numpy pandas
    pip install affine

.. _affine: https://github.com/sgillies/affine
.. _numpy: https://numpy.org/
.. _pandas: http://pandas.pydata.org

Installing Modflow-obs using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pip can be used to fetch Modflow-obs directly from GitHub:

.. code-block:: bash

    pip install git+git://github.com/aleaf/modflow-obs@master

Subsequent updates can then be made with

.. code-block:: bash

    pip uninstall modflow-obs
    pip install git+git://github.com/aleaf/modflow-obs@master

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
