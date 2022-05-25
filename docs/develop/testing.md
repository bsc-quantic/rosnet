# Testing

We use `pytest` for write unit tests.

## dataClay

Code involving dataClay needs a some steps to be performed previously:

> The `dataclaycmd` command is an alias for sending commands to a Docker instance. `rosnet` installs it as a script.


1. Bring up the dataClay services.
```bash
docker-compose -f docker-compose.test.yml
```

2. Create an account.
```bash
dataclaycmd NewAccount <USER> <SECRET>
dataclaycmd NewDataContract <USER> <SECRET> <DATASET> <USER>
```

3. Register the data model (i.e. the `DataClayBlock` class in `rosnet_dclaymodel`).
```bash
dataclaycmd NewModel <USER> <SECRET> rosnet_dclaymodel /workdir/rosnet_dclaymodel python
dataclaycmd GetStubs <USER> <SECRET> rosnet_dclaymodel /workdir/stubs
```

4. Install the `dataclay` client library.

### Known issues

#### `AttributeError: module 'collections' has no attribute 'MutableMapping'`
Python 3.10 deprecated `collections.MutableMapping`, needed for dataClay 2.7 or earlier. While the fix is solved, you can manually change edit line 11 in `dataclay/util/IdentityDict.py`:that commands

```python
...
class IdentityDict(collections.MutableMapping):
	...
```

for
```python
...
class IdentityDict(collections.abc.MutableMapping):
	...
```

#### `AttributeError: 'NoneType' object has no attribute 'add_to_heap'`

You have to init dataClay before using it.

```python
from dataclay.api import init

init()
```