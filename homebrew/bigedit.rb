class Bigedit < Formula
  desc "Fast text editor for very large files using journaling and FUSE"
  homepage "https://github.com/jopdorp/bigedit"
  url "https://github.com/jopdorp/bigedit/archive/refs/tags/v0.1.6.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"
  head "https://github.com/jopdorp/bigedit.git", branch: "master"

  depends_on "rust" => :build
  depends_on "pkg-config" => :build

  on_linux do
    depends_on "libfuse"
  end

  def install
    if OS.mac?
      # On macOS, check if macFUSE is installed
      if File.exist?("/Library/Filesystems/macfuse.fs") || File.exist?("/usr/local/lib/libfuse.dylib")
        # macFUSE is installed, build with FUSE support
        system "cargo", "install", *std_cargo_args
        system "cargo", "build", "--release", "--bin", "bigedit-fuse"
        bin.install "target/release/bigedit-fuse"
      else
        # No macFUSE, build without FUSE
        system "cargo", "install", *std_cargo_args, "--no-default-features"
      end
    else
      # On Linux, build with FUSE support
      system "cargo", "install", *std_cargo_args
      system "cargo", "build", "--release", "--bin", "bigedit-fuse"
      bin.install "target/release/bigedit-fuse"
    end
  end

  def caveats
    if OS.mac? && !File.exist?("/Library/Filesystems/macfuse.fs") && !File.exist?("/usr/local/lib/libfuse.dylib")
      <<~EOS
        bigedit was installed WITHOUT FUSE support.
        
        To enable FUSE features (virtual file view for other programs):
          1. Install macFUSE: brew install --cask macfuse
          2. Allow the kernel extension in System Settings > Privacy & Security
          3. Reinstall bigedit: brew reinstall bigedit
      EOS
    else
      <<~EOS
        Usage:
          bigedit <filename>

        Keyboard shortcuts (Nano mode, default):
          Ctrl+O  Save (journal mode - instant)
          Ctrl+J  Compact (rewrite entire file)  
          Ctrl+X  Exit
          Ctrl+Z  Undo
          Ctrl+Y  Redo
          F2      Toggle Vi mode
          Ctrl+T  Toggle FUSE mode

        FUSE support is enabled! Other programs can see your edits in real-time.
      EOS
    end
  end

  test do
    (testpath/"test.txt").write("Hello, World!")
    assert_match version.to_s, shell_output("#{bin}/bigedit --version 2>&1", 1)
  end
end
