inpath = ARGV[0]

require 'rubygems'

require 'liquid'
require 'yaml'
require_relative '_plugins/idio'
require_relative '_plugins/figurepath'
require 'ostruct'

# Liquid::Template.error_mode = :strict Removed for older liquid compatibility

content = File.read(inpath)

if ARGV[1] == "slides"
  slides = true
  latex = false
else
  slides = false
  latex = true
end

front = YAML.load(content[/\A---(.|\n)*?---/])

if latex
  content.gsub!(/\A---(.|\n)*?---/,'')
  if inpath.include?("index.md")
    content="\n\n# #{front["title"]}\n" + content
  end
end

@template = Liquid::Template.parse(content)

site = OpenStruct.new
site.config={
  "idio" => {
    "fence" => true
  },
  "absolute_figures" => latex,
  "latex" => latex,
  "slides" => slides
}

page = {
  "path" => inpath
}

registers = {
  :site => site,
  :page => page
}

context = Liquid::Context.new({}, {}, registers)

print @template.render(context)
