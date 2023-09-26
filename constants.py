EMAIL_TEMPLATE = """
Hi,

Thank you for your enquiry. You may be interested in:

INSERT_PROPERTIES

If you have any further questions, feel free to email or call. Have a great \
    day!

Many thanks,

Comprende Property Management Ltd.
"""


PROMPT_TEMPLATE = """
You work for a property management company, Comprende Ltd. You have been given 
an email from a prospective tenant, enquiring about whether we have rental 
properties to meet their needs, and a full list of Comprende's rental 
properties.

Your task is to match the tenant to the right properties.

Here is the tenant's email: {tenant_email}

Here are Comprende's rental properties: {properties}

Please write a polite, helpful email in response.
"""

