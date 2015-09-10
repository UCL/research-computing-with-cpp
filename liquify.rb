inpath = ARGV[0]

require 'rubygems'

require 'liquid'
require_relative '_plugins/idio'
require_relative '_plugins/figurepath'
require 'ostruct'

Liquid::Template.error_mode = :strict

@template = Liquid::Template.parse(File.read(inpath))

site = OpenStruct.new
site.config={
  "idio" => {
    "fence" => true
  },
  "absolute_figures" => true
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
