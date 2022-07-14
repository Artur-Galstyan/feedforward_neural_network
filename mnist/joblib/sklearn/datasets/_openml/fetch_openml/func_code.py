# first line: 590
def fetch_openml(
    name: Optional[str] = None,
    *,
    version: Union[str, int] = "active",
    data_id: Optional[int] = None,
    data_home: Optional[str] = None,
    target_column: Optional[Union[str, List]] = "default-target",
    cache: bool = True,
    return_X_y: bool = False,
    as_frame: Union[str, bool] = "auto",
    n_retries: int = 3,
    delay: float = 1.0,
):
    """Fetch dataset from openml by name or dataset id.

    Datasets are uniquely identified by either an integer ID or by a
    combination of name and version (i.e. there might be multiple
    versions of the 'iris' dataset). Please give either name or data_id
    (not both). In case a name is given, a version can also be
    provided.

    Read more in the :ref:`User Guide <openml>`.

    .. versionadded:: 0.20

    .. note:: EXPERIMENTAL

        The API is experimental (particularly the return value structure),
        and might have small backward-incompatible changes without notice
        or warning in future releases.

    Parameters
    ----------
    name : str, default=None
        String identifier of the dataset. Note that OpenML can have multiple
        datasets with the same name.

    version : int or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.

    data_id : int, default=None
        OpenML ID of the dataset. The most specific way of retrieving a
        dataset. If data_id is not given, name (and potential version) are
        used to obtain a dataset.

    data_home : str, default=None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    target_column : str, list or None, default='default-target'
        Specify the column name in the data to use as target. If
        'default-target', the standard target column a stored on the server
        is used. If ``None``, all columns are returned as data and the
        target is ``None``. If list (of strings), all columns with these names
        are returned as multi-target (Note: not all scikit-learn classifiers
        can handle all types of multi-output combinations).

    cache : bool, default=True
        Whether to cache the downloaded datasets into `data_home`.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` objects.

    as_frame : bool or 'auto', default='auto'
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

        If as_frame is 'auto', the data and target will be converted to
        DataFrame or Series as if as_frame is set to True, unless the dataset
        is stored in sparse format.

        .. versionchanged:: 0.24
           The default value of `as_frame` changed from `False` to `'auto'`
           in 0.24.

    n_retries : int, default=3
        Number of retries when HTTP errors or network timeouts are encountered.
        Error with status code 412 won't be retried as they represent OpenML
        generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------

    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : np.array, scipy.sparse.csr_matrix of floats, or pandas DataFrame
            The feature matrix. Categorical features are encoded as ordinals.
        target : np.array, pandas Series or DataFrame
            The regression target or classification labels, if applicable.
            Dtype is float if numeric, and object if categorical. If
            ``as_frame`` is True, ``target`` is a pandas object.
        DESCR : str
            The full description of the dataset.
        feature_names : list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.

        .. versionadded:: 0.22

        categories : dict or None
            Maps each categorical feature name to a list of values, such
            that the value encoded as i is ith in the list. If ``as_frame``
            is True, this is None.
        details : dict
            More metadata from OpenML.
        frame : pandas DataFrame
            Only present when `as_frame=True`. DataFrame with ``data`` and
            ``target``.

    (data, target) : tuple if ``return_X_y`` is True

        .. note:: EXPERIMENTAL

            This interface is **experimental** and subsequent releases may
            change attributes without notice (although there should only be
            minor changes to ``data`` and ``target``).

        Missing values in the 'data' are represented as NaN's. Missing values
        in 'target' are represented as NaN's (numerical target) or None
        (categorical target).
    """
    if cache is False:
        # no caching will be applied
        data_home = None
    else:
        data_home = get_data_home(data_home=data_home)
        data_home = join(data_home, "openml")

    # check valid function arguments. data_id XOR (name, version) should be
    # provided
    if name is not None:
        # OpenML is case-insensitive, but the caching mechanism is not
        # convert all data names (str) to lower case
        name = name.lower()
        if data_id is not None:
            raise ValueError(
                "Dataset data_id={} and name={} passed, but you can only "
                "specify a numeric data_id or a name, not "
                "both.".format(data_id, name)
            )
        data_info = _get_data_info_by_name(
            name, version, data_home, n_retries=n_retries, delay=delay
        )
        data_id = data_info["did"]
    elif data_id is not None:
        # from the previous if statement, it is given that name is None
        if version != "active":
            raise ValueError(
                "Dataset data_id={} and version={} passed, but you can only "
                "specify a numeric data_id or a version, not "
                "both.".format(data_id, version)
            )
    else:
        raise ValueError(
            "Neither name nor data_id are provided. Please provide name or data_id."
        )

    data_description = _get_data_description_by_id(data_id, data_home)
    if data_description["status"] != "active":
        warn(
            "Version {} of dataset {} is inactive, meaning that issues have "
            "been found in the dataset. Try using a newer version from "
            "this URL: {}".format(
                data_description["version"],
                data_description["name"],
                data_description["url"],
            )
        )
    if "error" in data_description:
        warn(
            "OpenML registered a problem with the dataset. It might be "
            "unusable. Error: {}".format(data_description["error"])
        )
    if "warning" in data_description:
        warn(
            "OpenML raised a warning on the dataset. It might be "
            "unusable. Warning: {}".format(data_description["warning"])
        )

    return_sparse = False
    if data_description["format"].lower() == "sparse_arff":
        return_sparse = True

    if as_frame == "auto":
        as_frame = not return_sparse

    if as_frame and return_sparse:
        raise ValueError("Cannot return dataframe with sparse data")

    # download data features, meta-info about column types
    features_list = _get_data_features(data_id, data_home)

    if not as_frame:
        for feature in features_list:
            if "true" in (feature["is_ignore"], feature["is_row_identifier"]):
                continue
            if feature["data_type"] == "string":
                raise ValueError(
                    "STRING attributes are not supported for "
                    "array representation. Try as_frame=True"
                )

    if target_column == "default-target":
        # determines the default target based on the data feature results
        # (which is currently more reliable than the data description;
        # see issue: https://github.com/openml/OpenML/issues/768)
        target_columns = [
            feature["name"]
            for feature in features_list
            if feature["is_target"] == "true"
        ]
    elif isinstance(target_column, str):
        # for code-simplicity, make target_column by default a list
        target_columns = [target_column]
    elif target_column is None:
        target_columns = []
    elif isinstance(target_column, list):
        target_columns = target_column
    else:
        raise TypeError(
            "Did not recognize type of target_column"
            "Should be str, list or None. Got: "
            "{}".format(type(target_column))
        )
    data_columns = _valid_data_column_names(features_list, target_columns)

    shape: Optional[Tuple[int, int]]
    # determine arff encoding to return
    if not return_sparse:
        # The shape must include the ignored features to keep the right indexes
        # during the arff data conversion.
        data_qualities = _get_data_qualities(data_id, data_home)
        shape = _get_num_samples(data_qualities), len(features_list)
    else:
        shape = None

    # obtain the data
    url = _DATA_FILE.format(data_description["file_id"])
    bunch = _download_data_to_bunch(
        url,
        return_sparse,
        data_home,
        as_frame=bool(as_frame),
        features_list=features_list,
        shape=shape,
        target_columns=target_columns,
        data_columns=data_columns,
        md5_checksum=data_description["md5_checksum"],
        n_retries=n_retries,
        delay=delay,
    )

    if return_X_y:
        return bunch.data, bunch.target

    description = "{}\n\nDownloaded from openml.org.".format(
        data_description.pop("description")
    )

    bunch.update(
        DESCR=description,
        details=data_description,
        url="https://www.openml.org/d/{}".format(data_id),
    )

    return bunch
