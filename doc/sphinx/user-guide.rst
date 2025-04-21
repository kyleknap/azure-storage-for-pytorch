User Guide
==========

.. _getting-started:

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~
* Python 3.9 or later installed
* Have an `Azure subscription`_ and an `Azure storage account`_

Installation
~~~~~~~~~~~~
Install the Azure Storage for PyTorch (``azstoragetorch``) library with `pip`_:

.. code-block:: shell

    pip install azstoragetorch


Configuration
~~~~~~~~~~~~~

``azstoragetorch`` should work without any explicit credential configuration.

``azstoragetorch`` interfaces default to :py:class:`~azure.identity.DefaultAzureCredential`
for  credentials. ``DefaultAzureCredential`` automatically retrieves
`Microsoft Entra ID tokens`_ based on your current environment. For more information
on ``DefaultAzureCredential``, see its `documentation <DefaultAzureCredential guide_>`_.

To override credentials, ``azstoragetorch`` interfaces accept a ``credential``
keyword argument override and accept `SAS`_ tokens in query strings of
provided Azure Storage URLs. See the :doc:`API Reference <api>` for more details.


.. _checkpoint-guide:

Saving and Loading PyTorch Models (Checkpointing)
-------------------------------------------------

PyTorch `supports saving and loading trained models <PyTorch checkpoint tutorial_>`_
(i.e., checkpointing). The core PyTorch interfaces for saving and loading models are
:py:func:`torch.save` and :py:func:`torch.load` respectively. Both of these functions
accept a file-like object to be written to or read from.

``azstoragetorch`` offers the :py:class:`azstoragetorch.io.BlobIO` file-like object class
to save and load models directly to and from Azure Blob Storage when using :py:func:`torch.save`
and :py:func:`torch.load`.

Saving a Model
~~~~~~~~~~~~~~
To save a model to Azure Blob Storage, pass a :py:class:`azstoragetorch.io.BlobIO`
directly to :py:func:`torch.save`. When creating the :py:class:`~azstoragetorch.io.BlobIO`,
specify the URL to the blob you'd like to save the model to and use write mode (i.e., ``wb``)::

    import torch
    import torchvision.models  # Install separately: ``pip install torchvision``
    from azstoragetorch.io import BlobIO

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    # Model to save. Replace with your own model.
    model = torchvision.models.resnet18(weights="DEFAULT")

    # Save trained model to Azure Blob Storage. This saves the model weights
    # to a blob named "model_weights.pth" in the container specified by CONTAINER_URL.
    with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "wb") as f:
        torch.save(model.state_dict(), f)


Loading a Model
~~~~~~~~~~~~~~~
To load a model from Azure Blob Storage, pass a :py:class:`azstoragetorch.io.BlobIO`
directly to :py:func:`torch.load`. When creating the :py:class:`~azstoragetorch.io.BlobIO`,
specify the URL to the blob storing the model weights and use read mode (i.e., ``rb``)::

    import torch
    import torchvision.models  # Install separately: ``pip install torchvision``
    from azstoragetorch.io import BlobIO

    # Update URL with your own Azure Storage account and container name
    CONTAINER_URL = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"

    # Model to load weights for. Replace with your own model.
    model = torchvision.models.resnet18()

    # Load trained model from Azure Blob Storage.  This loads the model weights
    # from the blob named "model_weights.pth" in the container specified by CONTAINER_URL.
    with BlobIO(f"{CONTAINER_URL}/model_weights.pth", "rb") as f:
        model.load_state_dict(torch.load(f))


.. _Azure subscription: https://azure.microsoft.com/free/
.. _Azure storage account: https://learn.microsoft.com/azure/storage/common/storage-account-overview
.. _pip: https://pypi.org/project/pip/
.. _Microsoft Entra ID tokens: https://learn.microsoft.com/en-us/azure/storage/blobs/authorize-access-azure-active-directory
.. _DefaultAzureCredential guide: https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/credential-chains?tabs=dac#defaultazurecredential-overview
.. _SAS: https://learn.microsoft.com/en-us/azure/storage/common/storage-sas-overview
.. _PyTorch checkpoint tutorial: https://pytorch.org/tutorials/beginner/saving_loading_models.html
