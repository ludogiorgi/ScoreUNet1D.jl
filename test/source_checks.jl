using Test
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const TRACKED = readlines(`git -C $ROOT ls-files`)

@testset "KS-only public repository boundary" begin
    forbidden = [
        "AGENTS.md",
        "FINAL_SOURCES.md",
        "PROJECT.yaml",
        ".codex",
        "manuscript",
        "paper",
        "scripts/L96",
        "scripts/arnold",
    ]
    for relative_path in forbidden
        @test !ispath(joinpath(ROOT, relative_path))
    end
end

@testset "tracked Julia syntax" begin
    julia_files = filter(path -> endswith(path, ".jl") && isfile(joinpath(ROOT, path)), TRACKED)
    @test !isempty(julia_files)
    for relative_path in julia_files
        source = read(joinpath(ROOT, relative_path), String)
        @test Meta.parseall(source; filename=relative_path) isa Expr
    end
end

@testset "tracked TOML syntax" begin
    toml_files = filter(path -> endswith(path, ".toml") && isfile(joinpath(ROOT, path)), TRACKED)
    @test !isempty(toml_files)
    for relative_path in toml_files
        @test TOML.parsefile(joinpath(ROOT, relative_path)) isa Dict
    end
end

@testset "no private machine locations" begin
    markers = [
        join(("/", "Users", "/")),
        join(("/home/", "ludo", "gio")),
        join(("/data/", "ludo", "gio")),
        string("my", "remote"),
    ]
    text_extensions = (".jl", ".toml", ".md", ".yml", ".yaml", ".cff")
    text_files = filter(TRACKED) do path
        isfile(joinpath(ROOT, path)) && any(endswith(path, extension) for extension in text_extensions)
    end
    for relative_path in text_files
        contents = read(joinpath(ROOT, relative_path), String)
        for marker in markers
            @test !occursin(marker, contents)
        end
    end
end
