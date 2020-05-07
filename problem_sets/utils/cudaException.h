#include <stdint.h>
#include <string>

class cudaException
{
  public:
    explicit cudaException(const std::string& rMessage = "",
                           const std::string& rFileName = "",
                           unsigned int nLineNumber = 0)
      : sMessage_(rMessage)
      , sFileName_(rFileName)
      , nLineNumber_(nLineNumber){};

    cudaException(const cudaException& other)
      : sMessage_(other.sMessage_)
      , sFileName_(other.sFileName_)
      , nLineNumber_(other.nLineNumber_){};

    ~cudaException(){};

    /// Get the exception's message.
    const std::string& message() const { return message_; }

    /// Get the exception's file info.
    const std::string& fileName() const { return file_name_; }

    /// Get the exceptions's line info.
    uint32_t lineNumber() const { return line_number_; }

    /// Create a single string with all the exceptions information.
    ///     The virtual toString() method is used by the operator<<()
    /// so that all exceptions derived from this base-class can print
    /// their full information correctly even if a reference to their
    /// exact type is not had at the time of printing (i.e. the basic
    /// operator<<() is used).
    std::string toString() const
    {
        std::ostringstream oOutputString;
        oOutputString << fileName() << ":" << lineNumber() << ": " << message();
        return oOutputString.str();
    }

  private:
    std::string message_;
    std::string file_name_;
    unsigned int line_number_;
};