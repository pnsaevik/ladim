"""
Schema module
"""

from pydantic import BaseModel, Field, ConfigDict
import datetime
import copy


class SolverConfig(BaseModel):
    """
    Solver configuration
    """

    start: datetime.date = Field(..., description="Start time of simulation")


class GridConfig(BaseModel):
    """
    Grid configuration
    """

    file: str = Field(..., description="File name, see also :any:`output.file`")


def _descr(txt):
    def fn(s):
        s['additionalProperties']['description'] = txt
    return fn


class OutvarSpec(BaseModel):
    """
    Output variable format specification
    """
    model_config = ConfigDict(extra='allow', json_schema_extra=_descr("Additional attributes"))
    __pydantic_extra__: dict[str, str] = Field(..., description="Additional attributes")

    name: str = Field(..., description="Variable name")
    format: str = Field(..., description="Variable output format")


class OutputConfig(BaseModel):
    """
    Output configuration
    """
    model_config = ConfigDict(extra='allow', json_schema_extra=_descr("Output format specification"))
    __pydantic_extra__: dict[str, OutvarSpec] = Field(..., description="Output format specification")

    file: str = Field(..., description="File name")

class Configuration(BaseModel):
    """
    Ladim configuration

    """
    
    solver: SolverConfig = Field(..., description="Solver configurations")
    grid: GridConfig = Field(..., description="Grid configurations")
    output: OutputConfig = Field(..., description="Output configurations")
    version: int = Field(default=1, description="Version number of config spec")


def jsonschema() -> dict:
    """
    Return the configuration specification as json schema
    """
    return Configuration.model_json_schema()


def rest_doc() -> str:
    """
    Create schema documentation in ReST format based on docstrings

    :returns: ReST-formatted documentation
    """

    json_dict = jsonschema()
    flat_doc = extract_keyword_properties(json_dict)

    rst_txt = ".. role:: ladimdoc-key\n  :class: ladimdoc-key\n\n"
    rst_txt += ".. role:: ladimdoc-sig\n  :class: ladimdoc-sig\n\n"
    rst_txt += ".. role:: ladimdoc-dsc\n  :class: ladimdoc-dsc\n\n"
    for item in flat_doc:
        link_target = item['name'].replace('<', '').replace('>', '')
        rst_txt += f".. container:: ladimdoc\n"
        rst_txt += f"    :name: {link_target}\n\n"
        rst_txt += f"    :ladimdoc-key:`{item['name']}`"
        if not item['required']:
            rst_txt += f", :ladimdoc-sig:`default = {item['default']}`"
        rst_txt += f": {item['description']}"
        rst_txt += "\n\n"
    
    return rst_txt


def extract_keyword_properties(jsondict: dict, prefix="", subtype="") -> list[dict]:
    """
    Extract flat list of keywords with properties

    This function takes a json schema as input. It iterates through all
    properties and extract their name, description, data type and default
    values. If any property is an object type, it iterates also through the
    descriptions of the subtype.

    If there is an "additionalProperties" description, this is rendered as
    a regular attribute with the name "<props>. 

    Nested object types are un-nested and their parent attribute name is
    appended to the attribute name.
    
    :param jsondict: A valid json schema, where subtypes are defined in
        a ``$defs`` entry.
    :param prefix: Prefix to append to attribute names
    :param subtype: Subtype within the jsondict from which attributes should
        be returned
    :returns: A list of dicts, one entry for each attribute. Dict keys are:
        
        - ``name``: The attribute name, possibly prefixed with parent attributes
        - ``description``: Attribute description
        - ``required``: True if attribute is required
        - ``default``: Default value of attribute, None if no default value
        - ``type``: Text description of data type
    """
    items = []
    if subtype == "":
        root = jsondict
    else:
        root = jsondict['$defs'][subtype]
    
    if 'additionalProperties' in root:
        root = copy.deepcopy(root)
        root['properties']['<props>'] = root['additionalProperties']

    for k, v in root['properties'].items():
        display_name = prefix + k
        if '$ref' in v.keys():
            assert v['$ref'].startswith('#/$defs/')
            dtype = v['$ref'][8:]
        else:
            dtype = v.get('type', None)
        new_item = dict(
            name=display_name,
            description=v.get('description', ''),
            required=k in root.get('required', []),
            default=v.get('default', None),
            type=dtype,
        )
        items.append(new_item)

        if '$ref' in v.keys():
            items += extract_keyword_properties(
                jsondict, prefix=display_name + ".", subtype=dtype)

    return items
