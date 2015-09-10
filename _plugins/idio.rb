module Idio

  class Helper
    attr_accessor :language, :separator, :content, :fence

	  def initialize(context, path)

      @config=  context.registers[:site].config.fetch("idio",{})

      prefix = @config.fetch("path", nil)
      here = File.dirname(context.registers[:page]["path"])

	    if not path.nil?
        if not prefix.nil?
          @path = File.join(here, prefix, path)
        else
          @path = File.join(here, path)
        end
      else
        if not prefix.nil?
          @path = File.join(here, prefix)
        else
          raise SyntaxError.new("No idio file supplied")
        end
      end

      begin
        @content = File.read(@path)
      rescue Errno::ENOENT
        raise SyntaxError.new("Missing idio file #{@path}")
      end

      extension = File.extname(@path).sub('.','')

      case extension
          when "py"
            @language = "python"
            @separator = "###"
          when "rb"
            @language = "ruby"
            @separator = "###"
          when "cpp", 'h', 'hpp', 'cc', 'c', 'cu'
            @language = "cpp"
            @separator = "///"
          when "java"
            @language = "java"
            @separator = "///"
          when "js"
            @language = "javascript"
            @separator = "///"
          else
            @separator = "###"
            @language = ""
      end

      @fence = @config.fetch("fence", false)

    end

	  def section(name)

      file_section_expression = /^\s*#{@separator}\s*(.*)\s*$/

      raw_sections = @content.split(file_section_expression)

      raw_sections = ["Beginning"]+raw_sections

      sections= raw_sections.each_slice(2).map{ |k, v|
        [k.downcase.gsub('"','') , v] }.to_h

      if not sections.keys.include?(name)
        raise SyntaxError.new("No such idio section as #{name} in #{@path}. Available: #{sections.keys} ")
      end

      return sections[name]

	  end
  end

  class Context < Liquid::Block
    def initialize(tag_name, markup, tokens)
      super
      @path = markup.strip.gsub('"','')
    end

    def render(context)
   	  config=  context.registers[:site].config.fetch("idio",{})
      @old = config['path']
      @old = "" if @old.nil?
      config['path']=File.join(@old, @path)
      context.registers[:site].config["idio"]=config
      catch = super
      config['path']=@old
      context.registers[:site].config["idio"]=config
      return catch
    end

  end

  class FragmentTag < Liquid::Tag
    Syntax = /([^,]*)(?:\s*,\s*(.*))?/o

    def initialize(tag_name, markup, tokens)
      super

      if markup =~ Syntax
        if $2
          @path = $2.strip
        else
          @path = nil
        end

	      @label = $1.downcase.strip.gsub('"','')

      else
        raise SyntaxError.new("Syntax error in idio fragment parsing: #{markup}")
      end
    end

    def render(context)

      helper = Helper.new(context, @path)

      content = helper.section(@label)

      if helper.fence
        return "``` #{helper.language}\n#{content}\n```\n"
      else
        return content
      end

    end
  end

  class CodeTag < Liquid::Tag

    def initialize(tag_name, markup, tokens)
      super

      @path = markup.strip.gsub('"','')

    end

    def render(context)

      helper = Helper.new(context, @path)

      content = helper.content

      if helper.fence
        return "``` #{helper.language}\n#{content}\n```\n"
      else
        return content
      end

    end
  end
end

Liquid::Template.register_tag('idio', Idio::Context)
Liquid::Template.register_tag('fragment', Idio::FragmentTag)
Liquid::Template.register_tag('code', Idio::CodeTag)
