from django import template

register = template.Library()

@register.filter
def replace_underscore(value):
    """Replace underscores with spaces and capitalize each word"""
    if isinstance(value, str):
        return value.replace('_', ' ').title()
    return value

@register.filter
def safe_title(value):
    """Safely apply title case to a string"""
    if isinstance(value, str):
        return value.replace('_', ' ').title()
    return str(value).title() if value else ''
