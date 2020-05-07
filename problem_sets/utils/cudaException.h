#include <cstdint>
#include <sstream>
#include <string>

namespace cs344 {
class cudaException
{
  public:
    explicit cudaException(const std::string& message = "",
                           const std::string& file_name = "",
                           unsigned int line_number = 0)
      : message_(message)
      , file_name_(file_name)
      , line_number_(line_number){};

    cudaException(const cudaException& other)
      : message_(other.message_)
      , file_name_(other.file_name_)
      , line_number_(other.line_number_){};

    ~cudaException(){};

    const std::string& message() const { return message_; }

    const std::string& fileName() const { return file_name_; }

    uint32_t lineNumber() const { return line_number_; }

    std::string toString() const
    {
        std::ostringstream output_string;
        output_string << fileName() << "@" << lineNumber() << ": " << message();
        return output_string.str();
    }

  private:
    std::string message_;
    std::string file_name_;
    unsigned int line_number_;
};
} // namespace cs344
