Implementation Guide
====================

Within a class implementation, variables with leading underscores are used to signify that they are not intended to be part of the public API. 

If variables without the leading underscore are used, it typically means the following:

1. If we are setting the variable, it is likely implemented as a property that does the proper checks (bounds checking, type checking, etc.)
2. If we are accessing the variable, we are probably using the public API of an object that isn't implemented in the current class definition. This is a common pattern for classes that use composition
3. If the usage doesn't fall under the points above, it could be we were lazy and haven't implemented the proper checks on the variable yet

Coding Conventions
==================

In adherence with PEP 8, ``underscore_case`` is used for variables and class definitions. ``TitleCase`` signifies a class. ``lowerCamelCase`` is almost ways avoided, except where it's the convention, such as for Qt programming. Mixing cases e.g.

.. code-block:: python

    mehData_fromExperiment

is a no-no.