# Contributing to landau deGUI {#contrib}

Contributions are welcomed via pull requests on the repository homepage! Before beginning work and
submitting a pull request, please contact the lead developer (currently Daniel Sussman) to see
if your intended feature is under development by anyone else at the moment and to make sure your plans
match up well with the existing code base.

# Currently planned features

Lots of stuff! We are particularly keen on adding additional model and updater classes to handle
active liquid crystals, so that seems like a likely next target. If you are interested in contributing
to any of these development branches please contact the lead developer!

# Source code conventions

## Coding

Code should be written in a style consistent with the existing code base. As a brief summary, the
Whitesmith indentation style should be used, and 4 spaces, and not tabs, should be used to indent
lines. A soft maximum line length of 120 characters should be used, with very long lines of code
broken at some natural point.

Variable names should be descriptive; prefer lowerCamelCase names to other options.
When using ArrayHandles stick to the h_variableName and d_variableName convention for accessing GPUArrays
on either the host or device.

Rules are meant to be broken when doing so improves readability or comprehensibility. But they're not meant to be broken too much.

## Documentation

Every class, member, function, etc., should be documented with doxygen comments. Thanks!
