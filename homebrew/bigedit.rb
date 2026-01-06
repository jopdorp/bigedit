class Bigedit < Formula
  desc "Fast text editor for very large files using journaling and FUSE"
  homepage "https://github.com/jopdorp/bigedit"
  url "https://github.com/jopdorp/bigedit/archive/refs/tags/v0.1.4.tar.gz"
  sha256 "ed8ed21d86820dd0606ffd2d074c98696dadccdd2b12ee10e0c897712b912f8a"
  license "MIT"
  head "https://github.com/jopdorp/bigedit.git", branch: "master"

  depends_on "rust" => :build
  depends_on "macfuse"

  def install
    system "cargo", "install", *std_cargo_args
    
    # Install the FUSE daemon
    system "cargo", "build", "--release", "--bin", "bigedit-fuse"
    bin.install "target/release/bigedit-fuse"
  end

  def caveats
    <<~EOS
      bigedit requires macFUSE to be installed for the FUSE features to work.
      
      After installing macFUSE, you may need to allow its kernel extension
      in System Preferences > Security & Privacy > Privacy & Security.
      
      Usage:
        bigedit <filename>
      
      Press Ctrl+T to toggle FUSE mode while editing.
    EOS
  end

  test do
    (testpath/"test.txt").write("Hello, World!")
    assert_match version.to_s, shell_output("#{bin}/bigedit --version 2>&1", 1)
  end
end
