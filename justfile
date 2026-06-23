fmt:
    rg --files -g '!{3pty}' -tcuda -tc -tcpp | xargs clang-format -style=file -Werror -i
